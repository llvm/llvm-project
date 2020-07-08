//===-- lib/Semantics/compute-offsets.cpp -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "compute-offsets.h"
#include "../../runtime/descriptor.h"
#include "flang/Evaluate/fold-designator.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/shape.h"
#include "flang/Evaluate/type.h"
#include "flang/Semantics/scope.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include "flang/Semantics/type.h"
#include <algorithm>
#include <vector>

namespace Fortran::semantics {

class ComputeOffsetsHelper {
public:
  // TODO: configure based on target
  static constexpr std::size_t maxAlignment{8};

  ComputeOffsetsHelper(SemanticsContext &context) : context_{context} {}
  void Compute() { Compute(context_.globalScope()); }

private:
  struct SizeAndAlignment {
    SizeAndAlignment() {}
    SizeAndAlignment(std::size_t bytes) : size{bytes}, alignment{bytes} {}
    SizeAndAlignment(std::size_t bytes, std::size_t align)
        : size{bytes}, alignment{align} {}
    std::size_t size{0};
    std::size_t alignment{0};
  };
  struct SymbolAndOffset {
    SymbolAndOffset(Symbol &s, std::size_t off, const EquivalenceObject &obj)
        : symbol{&s}, offset{off}, object{&obj} {}
    SymbolAndOffset(const SymbolAndOffset &) = default;
    Symbol *symbol;
    std::size_t offset;
    const EquivalenceObject *object;
  };

  void Compute(Scope &);
  void DoScope(Scope &);
  void DoCommonBlock(Symbol &);
  void DoEquivalenceSet(const EquivalenceSet &);
  SymbolAndOffset Resolve(const SymbolAndOffset &);
  std::size_t ComputeOffset(const EquivalenceObject &);
  void DoSymbol(Symbol &);
  SizeAndAlignment GetSizeAndAlignment(const Symbol &);
  SizeAndAlignment GetElementSize(const Symbol &);
  std::size_t CountElements(const Symbol &);
  static std::size_t Align(std::size_t, std::size_t);
  static SizeAndAlignment GetIntrinsicSizeAndAlignment(TypeCategory, int);

  SemanticsContext &context_;
  evaluate::FoldingContext &foldingContext_{context_.foldingContext()};
  std::size_t offset_{0};
  std::size_t alignment_{0};
  // symbol -> symbol+offset that determines its location, from EQUIVALENCE
  std::map<MutableSymbolRef, SymbolAndOffset> dependents_;
};

void ComputeOffsetsHelper::Compute(Scope &scope) {
  for (Scope &child : scope.children()) {
    Compute(child);
  }
  DoScope(scope);
}

static bool InCommonBlock(const Symbol &symbol) {
  const auto *details{symbol.detailsIf<ObjectEntityDetails>()};
  return details && details->commonBlock();
}

void ComputeOffsetsHelper::DoScope(Scope &scope) {
  if (scope.symbol() && scope.IsParameterizedDerivedType()) {
    return; // only process instantiations of parameterized derived types
  }
  // Symbols in common block get offsets from the beginning of the block
  for (auto &pair : scope.commonBlocks()) {
    DoCommonBlock(*pair.second);
  }
  // Build dependents_ from equivalences: symbol -> symbol+offset
  for (const EquivalenceSet &set : scope.equivalenceSets()) {
    DoEquivalenceSet(set);
  }
  offset_ = 0;
  alignment_ = 0;
  for (auto &symbol : scope.GetSymbols()) {
    if (!InCommonBlock(*symbol) &&
        dependents_.find(symbol) == dependents_.end()) {
      DoSymbol(*symbol);
    }
  }
  for (auto &[symbol, dep] : dependents_) {
    if (symbol->size() == 0) {
      SizeAndAlignment s{GetSizeAndAlignment(*symbol)};
      symbol->set_size(s.size);
      SymbolAndOffset resolved{Resolve(dep)};
      symbol->set_offset(dep.symbol->offset() + resolved.offset);
      offset_ = std::max(offset_, symbol->offset() + symbol->size());
    }
  }
  scope.set_size(offset_);
  scope.set_alignment(alignment_);
}

auto ComputeOffsetsHelper::Resolve(const SymbolAndOffset &dep)
    -> SymbolAndOffset {
  auto it{dependents_.find(*dep.symbol)};
  if (it == dependents_.end()) {
    return dep;
  } else {
    SymbolAndOffset result{Resolve(it->second)};
    result.offset += dep.offset;
    result.object = dep.object;
    return result;
  }
}

void ComputeOffsetsHelper::DoCommonBlock(Symbol &commonBlock) {
  auto &details{commonBlock.get<CommonBlockDetails>()};
  offset_ = 0;
  alignment_ = 0;
  for (auto &object : details.objects()) {
    DoSymbol(*object);
  }
  commonBlock.set_size(offset_);
  details.set_alignment(alignment_);
}

void ComputeOffsetsHelper::DoEquivalenceSet(const EquivalenceSet &set) {
  std::vector<SymbolAndOffset> symbolOffsets;
  std::optional<std::size_t> representative;
  for (const EquivalenceObject &object : set) {
    std::size_t offset{ComputeOffset(object)};
    SymbolAndOffset resolved{
        Resolve(SymbolAndOffset{object.symbol, offset, object})};
    symbolOffsets.push_back(resolved);
    if (!representative ||
        resolved.offset >= symbolOffsets[*representative].offset) {
      // The equivalenced object with the largest offset from its resolved
      // symbol will be the representative of this set, since the offsets
      // of the other objects will be positive relative to it.
      representative = symbolOffsets.size() - 1;
    }
  }
  CHECK(representative);
  const SymbolAndOffset &base{symbolOffsets[*representative]};
  for (const auto &[symbol, offset, object] : symbolOffsets) {
    if (symbol == base.symbol) {
      if (offset != base.offset) {
        auto x{evaluate::OffsetToDesignator(
            context_.foldingContext(), *symbol, base.offset, 1)};
        auto y{evaluate::OffsetToDesignator(
            context_.foldingContext(), *symbol, offset, 1)};
        if (x && y) {
          context_
              .Say(base.object->source,
                  "'%s' and '%s' cannot have the same first storage unit"_err_en_US,
                  x->AsFortran(), y->AsFortran())
              .Attach(object->source, "Incompatible reference to '%s'"_en_US,
                  y->AsFortran());
        } else { // error recovery
          context_
              .Say(base.object->source,
                  "'%s' (offset %zd bytes and %zd bytes) cannot have the same first storage unit"_err_en_US,
                  symbol->name(), base.offset, offset)
              .Attach(object->source,
                  "Incompatible reference to '%s' offset %zd bytes"_en_US,
                  symbol->name(), offset);
        }
      }
    } else {
      dependents_.emplace(*symbol,
          SymbolAndOffset{*base.symbol, base.offset - offset, *object});
    }
  }
}

// Offset of this equivalence object from the start of its variable.
std::size_t ComputeOffsetsHelper::ComputeOffset(
    const EquivalenceObject &object) {
  std::size_t offset{0};
  if (!object.subscripts.empty()) {
    const ArraySpec &shape{object.symbol.get<ObjectEntityDetails>().shape()};
    auto lbound{[&](std::size_t i) {
      return *ToInt64(shape[i].lbound().GetExplicit());
    }};
    auto ubound{[&](std::size_t i) {
      return *ToInt64(shape[i].ubound().GetExplicit());
    }};
    for (std::size_t i{object.subscripts.size() - 1};;) {
      offset += object.subscripts[i] - lbound(i);
      if (i == 0) {
        break;
      }
      --i;
      offset *= ubound(i) - lbound(i) + 1;
    }
  }
  auto result{offset * GetElementSize(object.symbol).size};
  if (object.substringStart) {
    int kind{context_.defaultKinds().GetDefaultKind(TypeCategory::Character)};
    if (const DeclTypeSpec * type{object.symbol.GetType()}) {
      if (const IntrinsicTypeSpec * intrinsic{type->AsIntrinsic()}) {
        kind = ToInt64(intrinsic->kind()).value_or(kind);
      }
    }
    result += kind * (*object.substringStart - 1);
  }
  return result;
}

void ComputeOffsetsHelper::DoSymbol(Symbol &symbol) {
  if (symbol.has<TypeParamDetails>() || symbol.has<SubprogramDetails>() ||
      symbol.has<UseDetails>() || symbol.has<ProcBindingDetails>()) {
    return; // these have type but no size
  }
  SizeAndAlignment s{GetSizeAndAlignment(symbol)};
  if (s.size == 0) {
    return;
  }
  offset_ = Align(offset_, s.alignment);
  symbol.set_size(s.size);
  symbol.set_offset(offset_);
  offset_ += s.size;
  alignment_ = std::max(alignment_, s.alignment);
}

auto ComputeOffsetsHelper::GetSizeAndAlignment(const Symbol &symbol)
    -> SizeAndAlignment {
  SizeAndAlignment result{GetElementSize(symbol)};
  std::size_t elements{CountElements(symbol)};
  if (elements > 1) {
    result.size = Align(result.size, result.alignment);
  }
  result.size *= elements;
  return result;
}

auto ComputeOffsetsHelper::GetElementSize(const Symbol &symbol)
    -> SizeAndAlignment {
  const DeclTypeSpec *type{symbol.GetType()};
  if (!type) {
    return {};
  }
  // TODO: The size of procedure pointers is not yet known
  // and is independent of rank (and probably also the number
  // of length type parameters).
  if (IsDescriptor(symbol) || IsProcedure(symbol)) {
    int lenParams{0};
    if (const DerivedTypeSpec * derived{type->AsDerived()}) {
      lenParams = CountLenParameters(*derived);
    }
    std::size_t size{
        runtime::Descriptor::SizeInBytes(symbol.Rank(), false, lenParams)};
    return {size, maxAlignment};
  }
  SizeAndAlignment result;
  if (const IntrinsicTypeSpec * intrinsic{type->AsIntrinsic()}) {
    if (auto kind{ToInt64(intrinsic->kind())}) {
      result = GetIntrinsicSizeAndAlignment(intrinsic->category(), *kind);
    }
    if (type->category() == DeclTypeSpec::Character) {
      ParamValue length{type->characterTypeSpec().length()};
      CHECK(length.isExplicit()); // else should be descriptor
      if (MaybeIntExpr lengthExpr{length.GetExplicit()}) {
        if (auto lengthInt{ToInt64(*lengthExpr)}) {
          result.size *= *lengthInt;
        }
      }
    }
  } else if (const DerivedTypeSpec * derived{type->AsDerived()}) {
    if (derived->scope()) {
      result.size = derived->scope()->size();
      result.alignment = derived->scope()->alignment();
    }
  } else {
    DIE("not intrinsic or derived");
  }
  return result;
}

std::size_t ComputeOffsetsHelper::CountElements(const Symbol &symbol) {
  if (auto shape{GetShape(foldingContext_, symbol)}) {
    if (auto sizeExpr{evaluate::GetSize(std::move(*shape))}) {
      if (auto size{ToInt64(Fold(foldingContext_, std::move(*sizeExpr)))}) {
        return *size;
      }
    }
  }
  return 1;
}

// Align a size to its natural alignment, up to maxAlignment.
std::size_t ComputeOffsetsHelper::Align(std::size_t x, std::size_t alignment) {
  if (alignment > maxAlignment) {
    alignment = maxAlignment;
  }
  return (x + alignment - 1) & -alignment;
}

auto ComputeOffsetsHelper::GetIntrinsicSizeAndAlignment(
    TypeCategory category, int kind) -> SizeAndAlignment {
  if (category == TypeCategory::Character) {
    return {static_cast<std::size_t>(kind)};
  }
  std::optional<std::size_t> size{
      evaluate::DynamicType{category, kind}.MeasureSizeInBytes()};
  CHECK(size.has_value());
  if (category == TypeCategory::Complex) {
    return {*size, *size >> 1};
  } else {
    return {*size};
  }
}

void ComputeOffsets(SemanticsContext &context) {
  ComputeOffsetsHelper{context}.Compute();
}

} // namespace Fortran::semantics
