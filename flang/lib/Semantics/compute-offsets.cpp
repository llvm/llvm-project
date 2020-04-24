//===----------------------------------------------------------------------===//
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
  static constexpr int descriptorSize{3 * 8};
  static constexpr int maxAlignment{8};

  ComputeOffsetsHelper(SemanticsContext &context) : context_{context} {}
  void Compute() { Compute(context_.globalScope()); }

private:
  struct SizeAndAlign {
    SizeAndAlign() {}
    SizeAndAlign(std::size_t size) : size{size}, align{size} {}
    SizeAndAlign(std::size_t size, std::size_t align)
        : size{size}, align{align} {}
    std::size_t size{0};
    std::size_t align{0};
  };

  void Compute(Scope &);
  void DoScope(Scope &);
  void DoSymbol(Symbol &);
  SizeAndAlign GetSizeAndAlign(const Symbol &);
  std::size_t CountElements(const Symbol &);
  static std::size_t Align(std::size_t, std::size_t);
  static SizeAndAlign GetIntrinsicSizeAndAlign(TypeCategory, int);

  SemanticsContext &context_;
  evaluate::FoldingContext &foldingContext_{context_.foldingContext()};
  std::size_t offset_{0};
  std::size_t align_{0};
};

void ComputeOffsetsHelper::Compute(Scope &scope) {
  for (Scope &child : scope.children()) {
    Compute(child);
  }
  DoScope(scope);
}

void ComputeOffsetsHelper::DoScope(Scope &scope) {
  if (scope.symbol() && scope.IsParameterizedDerivedType()) {
    return; // only process instantiations of parameterized derived types
  }
  offset_ = 0;
  align_ = 0;
  for (auto symbol : scope.GetSymbols()) {
    if (!symbol->has<TypeParamDetails>() && !symbol->has<SubprogramDetails>()) {
      DoSymbol(*symbol);
    }
  }
  scope.set_size(offset_);
  scope.set_align(align_);
}

void ComputeOffsetsHelper::DoSymbol(Symbol &symbol) {
  SizeAndAlign s{GetSizeAndAlign(symbol)};
  if (s.size == 0) {
    return;
  }
  offset_ = Align(offset_, s.align);
  symbol.set_size(s.size);
  symbol.set_offset(offset_);
  offset_ += s.size;
  if (s.align > align_) {
    align_ = s.align;
  }
}

auto ComputeOffsetsHelper::GetSizeAndAlign(const Symbol &symbol)
    -> SizeAndAlign {
  const DeclTypeSpec *type{symbol.GetType()};
  if (!type) {
    return {};
  }
  if (IsDescriptor(symbol) || IsProcedure(symbol)) {
    int lenParams{0};
    if (const DerivedTypeSpec * derived{type->AsDerived()}) {
      lenParams = derived->NumLengthParameters();
    }
    std::size_t size{
        runtime::Descriptor::SizeInBytes(symbol.Rank(), false, lenParams)};
    return {size, maxAlignment};
  }
  SizeAndAlign result;
  if (const IntrinsicTypeSpec * intrinsic{type->AsIntrinsic()}) {
    if (auto kind{ToInt64(intrinsic->kind())}) {
      result = GetIntrinsicSizeAndAlign(intrinsic->category(), *kind);
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
      result.align = derived->scope()->align();
    }
  } else {
    DIE("not intrinsic or derived");
  }
  std::size_t elements{CountElements(symbol)};
  if (elements > 1) {
    result.size = Align(result.size, result.align);
  }
  result.size *= elements;
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

auto ComputeOffsetsHelper::GetIntrinsicSizeAndAlign(
    TypeCategory category, int kind) -> SizeAndAlign {
  // TODO: does kind==10 need special handling?
  std::size_t size{kind == 3 ? 2 : static_cast<std::size_t>(kind)};
  if (category == TypeCategory::Complex) {
    return {2 * size, size};
  } else {
    return {size};
  }
}

void ComputeOffsets(SemanticsContext &context) {
  ComputeOffsetsHelper{context}.Compute();
}

} // namespace Fortran::semantics
