//===-- lib/Evaluate/designate.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/fold-designator.h"
#include "flang/Semantics/tools.h"

namespace Fortran::evaluate {

DEFINE_DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(OffsetSymbol)

std::optional<OffsetSymbol> DesignatorFolder::FoldDesignator(
    const Symbol &symbol, ConstantSubscript which) {
  if (!getLastComponent_ && IsAllocatableOrPointer(symbol)) {
    // A pointer may appear as a DATA statement object if it is the
    // rightmost symbol in a designator and has no subscripts.
    // An allocatable may appear if its initializer is NULL().
    if (which > 0) {
      isEmpty_ = true;
    } else {
      return OffsetSymbol{symbol, symbol.size()};
    }
  } else if (symbol.has<semantics::ObjectEntityDetails>() &&
      !IsNamedConstant(symbol)) {
    if (auto type{DynamicType::From(symbol)}) {
      if (auto extents{GetConstantExtents(context_, symbol)}) {
        if (auto bytes{ToInt64(
                type->MeasureSizeInBytes(context_, GetRank(*extents) > 0))}) {
          OffsetSymbol result{symbol, static_cast<std::size_t>(*bytes)};
          if (which < GetSize(*extents)) {
            result.Augment(*bytes * which);
            return result;
          } else {
            isEmpty_ = true;
          }
        }
      }
    }
  }
  return std::nullopt;
}

std::optional<OffsetSymbol> DesignatorFolder::FoldDesignator(
    const ArrayRef &x, ConstantSubscript which) {
  const Symbol &array{x.base().GetLastSymbol()};
  if (auto type{DynamicType::From(array)}) {
    if (auto extents{GetConstantExtents(context_, array)}) {
      if (auto bytes{ToInt64(type->MeasureSizeInBytes(context_, true))}) {
        Shape lbs{GetLBOUNDs(context_, x.base())};
        if (auto lowerBounds{AsConstantExtents(context_, lbs)}) {
          std::optional<OffsetSymbol> result;
          if (!x.base().IsSymbol() &&
              x.base().GetComponent().base().Rank() > 0) {
            // A(:)%B(1) - apply elementNumber_ to base
            result = FoldDesignator(x.base(), which);
            which = 0;
          } else { // A(1)%B(:) - apply elementNumber_ to subscripts
            result = FoldDesignator(x.base(), 0);
          }
          if (!result) {
            return std::nullopt;
          }
          auto stride{*bytes};
          int dim{0};
          for (const Subscript &subscript : x.subscript()) {
            ConstantSubscript lower{lowerBounds->at(dim)};
            ConstantSubscript extent{extents->at(dim)};
            ConstantSubscript upper{lower + extent - 1};
            if (!common::visit(
                    common::visitors{
                        [&](const IndirectSubscriptIntegerExpr &expr) {
                          auto folded{
                              Fold(context_, common::Clone(expr.value()))};
                          if (auto value{UnwrapConstantValue<SubscriptInteger>(
                                  folded)}) {
                            CHECK(value->Rank() <= 1);
                            if (value->size() != 0) {
                              // Apply subscript, possibly vector-valued
                              auto quotient{which / value->size()};
                              auto remainder{which - value->size() * quotient};
                              ConstantSubscript at{
                                  value->values().at(remainder).ToInt64()};
                              if (at < lower || at > upper) {
                                isOutOfRange_ = true;
                              }
                              result->Augment((at - lower) * stride);
                              which = quotient;
                              return true;
                            } else {
                              isEmpty_ = true;
                            }
                          }
                          return false;
                        },
                        [&](const Triplet &triplet) {
                          auto start{ToInt64(Fold(context_,
                              triplet.lower().value_or(ExtentExpr{lower})))};
                          auto end{ToInt64(Fold(context_,
                              triplet.upper().value_or(ExtentExpr{upper})))};
                          auto step{ToInt64(Fold(context_, triplet.stride()))};
                          if (start && end && step) {
                            if (*step != 0) {
                              ConstantSubscript range{
                                  (*end - *start + *step) / *step};
                              if (range > 0) {
                                auto quotient{which / range};
                                auto remainder{which - range * quotient};
                                auto j{*start + remainder * *step};
                                result->Augment((j - lower) * stride);
                                which = quotient;
                                return true;
                              } else {
                                isEmpty_ = true;
                              }
                            }
                          }
                          return false;
                        },
                    },
                    subscript.u)) {
              return std::nullopt;
            }
            ++dim;
            stride *= extent;
          }
          if (which > 0) {
            isEmpty_ = true;
          } else {
            return result;
          }
        }
      }
    }
  }
  return std::nullopt;
}

std::optional<OffsetSymbol> DesignatorFolder::FoldDesignator(
    const Component &component, ConstantSubscript which) {
  const Symbol &comp{component.GetLastSymbol()};
  if (getLastComponent_) {
    return FoldDesignator(comp, which);
  } else {
    const DataRef &base{component.base()};
    std::optional<OffsetSymbol> baseResult, compResult;
    if (base.Rank() == 0) { // A%X(:) - apply "which" to component
      baseResult = FoldDesignator(base, 0);
      compResult = FoldDesignator(comp, which);
    } else { // A(:)%X - apply "which" to base
      baseResult = FoldDesignator(base, which);
      compResult = FoldDesignator(comp, 0);
    }
    if (baseResult && compResult) {
      OffsetSymbol result{baseResult->symbol(), compResult->size()};
      result.Augment(
          baseResult->offset() + compResult->offset() + comp.offset());
      return {std::move(result)};
    } else {
      return std::nullopt;
    }
  }
}

std::optional<OffsetSymbol> DesignatorFolder::FoldDesignator(
    const ComplexPart &z, ConstantSubscript which) {
  if (auto result{FoldDesignator(z.complex(), which)}) {
    result->set_size(result->size() >> 1);
    if (z.part() == ComplexPart::Part::IM) {
      result->Augment(result->size());
    }
    return result;
  } else {
    return std::nullopt;
  }
}

std::optional<OffsetSymbol> DesignatorFolder::FoldDesignator(
    const DataRef &dataRef, ConstantSubscript which) {
  return common::visit(
      [&](const auto &x) { return FoldDesignator(x, which); }, dataRef.u);
}

std::optional<OffsetSymbol> DesignatorFolder::FoldDesignator(
    const NamedEntity &entity, ConstantSubscript which) {
  return entity.IsSymbol() ? FoldDesignator(entity.GetLastSymbol(), which)
                           : FoldDesignator(entity.GetComponent(), which);
}

std::optional<OffsetSymbol> DesignatorFolder::FoldDesignator(
    const CoarrayRef &, ConstantSubscript) {
  return std::nullopt;
}

std::optional<OffsetSymbol> DesignatorFolder::FoldDesignator(
    const ProcedureDesignator &proc, ConstantSubscript which) {
  if (const Symbol * symbol{proc.GetSymbol()}) {
    if (const Component * component{proc.GetComponent()}) {
      return FoldDesignator(*component, which);
    } else if (which > 0) {
      isEmpty_ = true;
    } else {
      return FoldDesignator(*symbol, 0);
    }
  }
  return std::nullopt;
}

// Conversions of offset symbols (back) to Designators

// Reconstructs subscripts.
// "offset" is decremented in place to hold remaining component offset.
static std::optional<ArrayRef> OffsetToArrayRef(FoldingContext &context,
    NamedEntity &&entity, const Shape &shape, const DynamicType &elementType,
    ConstantSubscript &offset) {
  auto extents{AsConstantExtents(context, shape)};
  Shape lbs{GetRawLowerBounds(context, entity)};
  auto lower{AsConstantExtents(context, lbs)};
  auto elementBytes{ToInt64(elementType.MeasureSizeInBytes(context, true))};
  if (!extents || !lower || !elementBytes || *elementBytes <= 0) {
    return std::nullopt;
  }
  int rank{GetRank(shape)};
  CHECK(extents->size() == static_cast<std::size_t>(rank) &&
      lower->size() == extents->size());
  auto element{offset / static_cast<std::size_t>(*elementBytes)};
  std::vector<Subscript> subscripts;
  auto at{element};
  for (int dim{0}; dim + 1 < rank; ++dim) {
    auto extent{(*extents)[dim]};
    if (extent <= 0) {
      return std::nullopt;
    }
    auto quotient{at / extent};
    auto remainder{at - quotient * extent};
    subscripts.emplace_back(ExtentExpr{(*lower)[dim] + remainder});
    at = quotient;
  }
  // This final subscript might be out of range for use in error reporting.
  subscripts.emplace_back(ExtentExpr{(*lower)[rank - 1] + at});
  offset -= element * static_cast<std::size_t>(*elementBytes);
  return ArrayRef{std::move(entity), std::move(subscripts)};
}

// Maps an offset back to a component, when unambiguous.
static const Symbol *OffsetToUniqueComponent(
    const semantics::DerivedTypeSpec &spec, ConstantSubscript offset) {
  const Symbol *result{nullptr};
  if (const semantics::Scope * scope{spec.scope()}) {
    for (const auto &pair : *scope) {
      const Symbol &component{*pair.second};
      if (offset >= static_cast<ConstantSubscript>(component.offset()) &&
          offset < static_cast<ConstantSubscript>(
                       component.offset() + component.size())) {
        if (result) {
          return nullptr; // MAP overlap or error recovery
        }
        result = &component;
      }
    }
  }
  return result;
}

// Converts an offset into subscripts &/or component references.  Recursive.
// Any remaining offset is left in place in the "offset" reference argument.
static std::optional<DataRef> OffsetToDataRef(FoldingContext &context,
    NamedEntity &&entity, ConstantSubscript &offset, std::size_t size) {
  const Symbol &symbol{entity.GetLastSymbol()};
  if (IsAllocatableOrPointer(symbol)) {
    return entity.IsSymbol() ? DataRef{symbol}
                             : DataRef{std::move(entity.GetComponent())};
  }
  std::optional<DataRef> result;
  if (std::optional<DynamicType> type{DynamicType::From(symbol)}) {
    if (!type->IsUnlimitedPolymorphic()) {
      if (std::optional<Shape> shape{GetShape(context, symbol)}) {
        if (GetRank(*shape) > 0) {
          if (auto aref{OffsetToArrayRef(
                  context, std::move(entity), *shape, *type, offset)}) {
            result = DataRef{std::move(*aref)};
          }
        } else {
          result = entity.IsSymbol()
              ? DataRef{symbol}
              : DataRef{std::move(entity.GetComponent())};
        }
        if (result && type->category() == TypeCategory::Derived &&
            size < result->GetLastSymbol().size()) {
          if (const Symbol *
              component{OffsetToUniqueComponent(
                  type->GetDerivedTypeSpec(), offset)}) {
            offset -= component->offset();
            return OffsetToDataRef(context,
                NamedEntity{Component{std::move(*result), *component}}, offset,
                size);
          }
          result.reset();
        }
      }
    }
  }
  return result;
}

// Reconstructs a Designator from a symbol, an offset, and a size.
std::optional<Expr<SomeType>> OffsetToDesignator(FoldingContext &context,
    const Symbol &baseSymbol, ConstantSubscript offset, std::size_t size) {
  if (offset < 0) {
    return std::nullopt;
  }
  if (std::optional<DataRef> dataRef{
          OffsetToDataRef(context, NamedEntity{baseSymbol}, offset, size)}) {
    const Symbol &symbol{dataRef->GetLastSymbol()};
    if (std::optional<Expr<SomeType>> result{
            AsGenericExpr(std::move(*dataRef))}) {
      if (IsAllocatableOrPointer(symbol)) {
      } else if (auto type{DynamicType::From(symbol)}) {
        if (auto elementBytes{
                ToInt64(type->MeasureSizeInBytes(context, true))}) {
          if (auto *zExpr{std::get_if<Expr<SomeComplex>>(&result->u)}) {
            if (size * 2 > static_cast<std::size_t>(*elementBytes)) {
              return result;
            } else if (offset == 0 || offset * 2 == *elementBytes) {
              // Pick a COMPLEX component
              auto part{
                  offset == 0 ? ComplexPart::Part::RE : ComplexPart::Part::IM};
              return common::visit(
                  [&](const auto &z) -> std::optional<Expr<SomeType>> {
                    using PartType = typename ResultType<decltype(z)>::Part;
                    return AsGenericExpr(Designator<PartType>{ComplexPart{
                        ExtractDataRef(std::move(*zExpr)).value(), part}});
                  },
                  zExpr->u);
            }
          } else if (auto *cExpr{
                         std::get_if<Expr<SomeCharacter>>(&result->u)}) {
            if (offset > 0 || size != static_cast<std::size_t>(*elementBytes)) {
              // Select a substring
              return common::visit(
                  [&](const auto &x) -> std::optional<Expr<SomeType>> {
                    using T = typename std::decay_t<decltype(x)>::Result;
                    return AsGenericExpr(Designator<T>{
                        Substring{ExtractDataRef(std::move(*cExpr)).value(),
                            std::optional<Expr<SubscriptInteger>>{
                                1 + (offset / T::kind)},
                            std::optional<Expr<SubscriptInteger>>{
                                1 + ((offset + size - 1) / T::kind)}}});
                  },
                  cExpr->u);
            }
          }
        }
      }
      if (offset == 0) {
        return result;
      }
    }
  }
  return std::nullopt;
}

std::optional<Expr<SomeType>> OffsetToDesignator(
    FoldingContext &context, const OffsetSymbol &offsetSymbol) {
  return OffsetToDesignator(context, offsetSymbol.symbol(),
      offsetSymbol.offset(), offsetSymbol.size());
}

ConstantObjectPointer ConstantObjectPointer::From(
    FoldingContext &context, const Expr<SomeType> &expr) {
  auto extents{GetConstantExtents(context, expr)};
  CHECK(extents);
  std::optional<uint64_t> optElements{TotalElementCount(*extents)};
  CHECK(optElements);
  uint64_t elements{*optElements};
  CHECK(elements > 0);
  int rank{GetRank(*extents)};
  ConstantSubscripts at(rank, 1);
  ConstantObjectPointer::Dimensions dimensions(rank);
  for (int j{0}; j < rank; ++j) {
    dimensions[j].extent = (*extents)[j];
  }
  DesignatorFolder designatorFolder{context};
  const Symbol *symbol{nullptr};
  ConstantSubscript baseOffset{0};
  std::size_t elementSize{0};
  for (std::size_t j{0}; j < elements; ++j) {
    auto folded{designatorFolder.FoldDesignator(expr)};
    CHECK(folded);
    if (j == 0) {
      symbol = &folded->symbol();
      baseOffset = folded->offset();
      elementSize = folded->size();
    } else {
      CHECK(symbol == &folded->symbol());
      CHECK(elementSize == folded->size());
    }
    int twoDim{-1};
    for (int k{0}; k < rank; ++k) {
      if (at[k] == 2 && twoDim == -1) {
        twoDim = k;
      } else if (at[k] != 1) {
        twoDim = -2;
      }
    }
    if (twoDim >= 0) {
      // Exactly one subscript is a 2 and the rest are 1.
      dimensions[twoDim].byteStride = folded->offset() - baseOffset;
    }
    ConstantSubscript checkOffset{baseOffset};
    for (int k{0}; k < rank; ++k) {
      checkOffset += (at[k] - 1) * dimensions[twoDim].byteStride;
    }
    CHECK(checkOffset == folded->offset());
    CHECK(IncrementSubscripts(at, *extents) == (j + 1 < elements));
  }
  CHECK(!designatorFolder.FoldDesignator(expr));
  return ConstantObjectPointer{
      DEREF(symbol), elementSize, std::move(dimensions)};
}
} // namespace Fortran::evaluate
