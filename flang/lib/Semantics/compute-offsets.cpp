//===-- lib/Semantics/compute-offsets.cpp -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "compute-offsets.h"
#include "../../runtime/descriptor.h"
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
    Symbol *symbol{nullptr};
    std::size_t offset{0};
  };

  void Compute(Scope &);
  void DoScope(Scope &);
  void DoCommonBlock(Symbol &);
  void DoEquivalenceSet(EquivalenceSet &);
  std::size_t GetOffset(SymbolAndOffset &);
  std::size_t ComputeOffset(const EquivalenceObject &);
  void DoSymbol(Symbol &);
  SizeAndAlignment GetSizeAndAlignment(const Symbol &);
  SizeAndAlignment GetElementSize(const Symbol &, bool isSubstring = false);
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
  for (EquivalenceSet &set : scope.equivalenceSets()) {
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
      symbol->set_offset(GetOffset(dep));
      offset_ = std::max(offset_, symbol->offset() + symbol->size());
    }
  }
  scope.set_size(offset_);
  scope.set_alignment(alignment_);
}

std::size_t ComputeOffsetsHelper::GetOffset(SymbolAndOffset &dep) {
  auto it{dependents_.find(*dep.symbol)};
  if (it == dependents_.end()) {
    return dep.symbol->offset() + dep.offset;
  } else {
    return GetOffset(it->second) + dep.offset;
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

void ComputeOffsetsHelper::DoEquivalenceSet(EquivalenceSet &set) {
  std::vector<SymbolAndOffset> symbolOffsets;
  SymbolAndOffset max;
  for (EquivalenceObject &object : set) {
    std::size_t offset{ComputeOffset(object)};
    symbolOffsets.push_back({&object.symbol, offset});
    if (offset >= max.offset) {
      max.offset = offset;
      max.symbol = &object.symbol;
    }
  }
  CHECK(max.symbol);
  for (auto &[symbol, offset] : symbolOffsets) {
    if (symbol != max.symbol) {
      dependents_.emplace(
          *symbol, SymbolAndOffset{max.symbol, max.offset - offset});
    }
  }
}

// Offset of this equivalence object from the start of its variable.
std::size_t ComputeOffsetsHelper::ComputeOffset(
    const EquivalenceObject &object) {
  std::size_t offset{0};
  if (object.substringStart) {
    offset = *object.substringStart - 1;
  }
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
  return offset *
      GetElementSize(object.symbol, object.substringStart.has_value()).size;
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

auto ComputeOffsetsHelper::GetElementSize(
    const Symbol &symbol, bool isSubstring) -> SizeAndAlignment {
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
    if (!isSubstring && type->category() == DeclTypeSpec::Character) {
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
