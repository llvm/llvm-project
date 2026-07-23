//===-- lib/Evaluate/designator-path.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/designator-path.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/tools.h"
#include "llvm/Support/ErrorHandling.h"

namespace Fortran::evaluate {

static DesignatorRelation ComparePartSymbols(const Symbol *x, const Symbol *y) {
  if (x == y) {
    return DesignatorRelation::Equal;
  }
  if (!x) {
    return DesignatorRelation::Contains;
  }
  if (!y) {
    return DesignatorRelation::ContainedBy;
  }
  return DesignatorRelation::Disjoint;
}

static bool IsFullSubscriptList(const std::vector<Subscript> &subscripts) {
  if (subscripts.empty()) {
    return false;
  }
  for (const Subscript &subscript : subscripts) {
    const auto *triplet{std::get_if<Triplet>(&subscript.u)};
    if (!triplet || !DesignatorPath::IsFullTriplet(*triplet)) {
      return false;
    }
  }
  return true;
}

static bool IsFullSlicePart(const DesignatorPath::Part &part) {
  return !part.symbol && IsFullSubscriptList(part.subscripts);
}

static bool AreAllFullSliceParts(
    const std::vector<DesignatorPath::Part> &parts, std::size_t first) {
  for (std::size_t i{first}; i < parts.size(); ++i) {
    if (!IsFullSlicePart(parts[i])) {
      return false;
    }
  }
  return true;
}

std::optional<DesignatorPath::ConstantSubscriptRange>
DesignatorPath::GetConstantSubscriptRange(const Subscript &subscript) {
  // Surface syntax `x(i)` maps to a scalar Subscript that holds the integer
  // expression `i`, not a Triplet.
  if (const auto *expr{
          std::get_if<IndirectSubscriptIntegerExpr>(&subscript.u)}) {
    if (auto value{ToInt64(expr->value())}) {
      return ConstantSubscriptRange{*value, *value};
    }
  } else if (const auto *triplet{std::get_if<Triplet>(&subscript.u)}) {
    // Surface syntax `x(l:u)` maps to a Triplet with explicit lower and upper
    // bounds and an implicit stride of one. Syntax like `x(:u)`, `x(l:)`, and
    // `x(:)` maps to missing lower and/or upper bounds, so it is not a
    // constant finite range here.
    const auto *lowerExpr{triplet->GetLower()};
    const auto *upperExpr{triplet->GetUpper()};
    auto lower{lowerExpr ? ToInt64(*lowerExpr) : std::nullopt};
    auto upper{upperExpr ? ToInt64(*upperExpr) : std::nullopt};
    // Surface syntax `x(l:u:s)` maps to the same Triplet representation with a
    // non-optional stride expression.
    auto stride{ToInt64(triplet->GetStride())};
    if (lower && upper && stride && *stride == 1) {
      return ConstantSubscriptRange{*lower, *upper};
    }
  }
  return std::nullopt;
}

bool DesignatorPath::IsFullTriplet(const Triplet &triplet) {
  // Surface syntax `x(:)` maps to a Triplet with no lower or upper bound and
  // an implicit stride of one.
  auto stride{ToInt64(triplet.GetStride())};
  return !triplet.GetLower() && !triplet.GetUpper() && stride && *stride == 1;
}

DesignatorRelation DesignatorPath::CompareSubscripts(
    const Subscript &x, const Subscript &y) {
  if (x == y) {
    return DesignatorRelation::Equal;
  }
  const auto *xTriplet{std::get_if<Triplet>(&x.u)};
  const auto *yTriplet{std::get_if<Triplet>(&y.u)};
  if (xTriplet && IsFullTriplet(*xTriplet)) {
    if (yTriplet && IsFullTriplet(*yTriplet)) {
      return DesignatorRelation::Equal;
    }
    return DesignatorRelation::Contains;
  }
  if (yTriplet && IsFullTriplet(*yTriplet)) {
    return DesignatorRelation::ContainedBy;
  }
  auto xRange{GetConstantSubscriptRange(x)};
  auto yRange{GetConstantSubscriptRange(y)};
  if (!xRange || !yRange) {
    // Constant triplets with strides other than one cannot be represented as a
    // single range here, so they are treated as disjoint for now. This could be
    // made more precise by expanding constant triplets into index sets and
    // comparing those sets.
    return DesignatorRelation::Disjoint;
  }
  if (xRange->upper < yRange->lower || yRange->upper < xRange->lower) {
    return DesignatorRelation::Disjoint;
  }
  if (xRange->lower == yRange->lower && xRange->upper == yRange->upper) {
    return DesignatorRelation::Equal;
  }
  if (xRange->lower <= yRange->lower && xRange->upper >= yRange->upper) {
    return DesignatorRelation::Contains;
  }
  if (yRange->lower <= xRange->lower && yRange->upper >= xRange->upper) {
    return DesignatorRelation::ContainedBy;
  }
  return DesignatorRelation::Overlaps;
}

bool DesignatorPath::SubscriptMayContain(
    const Subscript &x, const Subscript &y) {
  if (x == y) {
    return true;
  }
  const auto *xTriplet{std::get_if<Triplet>(&x.u)};
  const auto *yTriplet{std::get_if<Triplet>(&y.u)};
  if (xTriplet && IsFullTriplet(*xTriplet)) {
    return true;
  }
  if (yTriplet && IsFullTriplet(*yTriplet)) {
    return false;
  }
  if (!xTriplet && yTriplet) {
    return false;
  }
  auto xRange{GetConstantSubscriptRange(x)};
  auto yRange{GetConstantSubscriptRange(y)};
  if (xRange && yRange) {
    return xRange->lower <= yRange->lower && xRange->upper >= yRange->upper;
  }
  if (xTriplet) {
    return true;
  }
  return !ToInt64(std::get<IndirectSubscriptIntegerExpr>(x.u).value()) ||
      !ToInt64(std::get<IndirectSubscriptIntegerExpr>(y.u).value());
}

bool DesignatorPath::SubscriptListMayContain(
    const std::vector<Subscript> &x, const std::vector<Subscript> &y) {
  if (x.empty()) {
    return true;
  }
  if (IsFullSubscriptList(x)) {
    return y.empty() || x.size() == y.size();
  }
  if (y.empty()) {
    return false;
  }
  if (x.size() != y.size()) {
    return false;
  }
  for (std::size_t i{0}; i < x.size(); ++i) {
    if (!SubscriptMayContain(x[i], y[i])) {
      return false;
    }
  }
  return true;
}

bool DesignatorPath::PartMayContain(const Part &x, const Part &y) {
  return SubscriptListMayContain(x.subscripts, y.subscripts) &&
      (!x.symbol || x.symbol == y.symbol);
}

DesignatorRelation DesignatorPath::CombineRelations(
    bool contains, bool containedBy, bool overlaps) {
  if (overlaps || (contains && containedBy)) {
    return DesignatorRelation::Overlaps;
  }
  if (contains) {
    return DesignatorRelation::Contains;
  }
  if (containedBy) {
    return DesignatorRelation::ContainedBy;
  }
  return DesignatorRelation::Equal;
}

DesignatorRelation DesignatorPath::CompareSubscriptLists(
    const std::vector<Subscript> &x, const std::vector<Subscript> &y) {
  if (x.empty() && y.empty()) {
    return DesignatorRelation::Equal;
  }
  if (x.empty()) {
    return IsFullSubscriptList(y) ? DesignatorRelation::Equal
                                  : DesignatorRelation::Contains;
  }
  if (y.empty()) {
    return IsFullSubscriptList(x) ? DesignatorRelation::Equal
                                  : DesignatorRelation::ContainedBy;
  }
  const bool xFull{IsFullSubscriptList(x)};
  const bool yFull{IsFullSubscriptList(y)};
  if (xFull || yFull) {
    if (x.size() != y.size()) {
      return DesignatorRelation::Disjoint;
    }
    if (xFull && yFull) {
      return DesignatorRelation::Equal;
    }
    return xFull ? DesignatorRelation::Contains
                 : DesignatorRelation::ContainedBy;
  }
  if (x.size() != y.size()) {
    return DesignatorRelation::Disjoint;
  }
  bool contains{false};
  bool containedBy{false};
  bool overlaps{false};
  for (std::size_t i{0}; i < x.size(); ++i) {
    switch (CompareSubscripts(x[i], y[i])) {
    case DesignatorRelation::Equal:
      break;
    case DesignatorRelation::Contains:
      contains = true;
      break;
    case DesignatorRelation::ContainedBy:
      containedBy = true;
      break;
    case DesignatorRelation::Overlaps:
      overlaps = true;
      break;
    case DesignatorRelation::Disjoint:
      return DesignatorRelation::Disjoint;
    }
  }
  return CombineRelations(contains, containedBy, overlaps);
}

DesignatorRelation DesignatorPath::CompareParts(const Part &x, const Part &y) {
  DesignatorRelation subscriptRelation{
      CompareSubscriptLists(x.subscripts, y.subscripts)};
  if (subscriptRelation == DesignatorRelation::Disjoint) {
    return DesignatorRelation::Disjoint;
  }
  DesignatorRelation symbolRelation{ComparePartSymbols(x.symbol, y.symbol)};
  if (symbolRelation == DesignatorRelation::Disjoint) {
    return DesignatorRelation::Disjoint;
  }
  bool contains{subscriptRelation == DesignatorRelation::Contains ||
      symbolRelation == DesignatorRelation::Contains};
  bool containedBy{subscriptRelation == DesignatorRelation::ContainedBy ||
      symbolRelation == DesignatorRelation::ContainedBy};
  bool overlaps{subscriptRelation == DesignatorRelation::Overlaps ||
      symbolRelation == DesignatorRelation::Overlaps};
  return CombineRelations(contains, containedBy, overlaps);
}

DesignatorRelation DesignatorPath::Compare(const DesignatorPath &that) const {
  if (*this == that) {
    return DesignatorRelation::Equal;
  }
  if (empty() || that.empty()) {
    return DesignatorRelation::Disjoint;
  }
  if (base || that.base) {
    if (!base || !that.base || !(*base == *that.base)) {
      return DesignatorRelation::Disjoint;
    }
  }
  if (parts.empty() || that.parts.empty()) {
    if ((!parts.empty() && AreAllFullSliceParts(parts, 0)) ||
        (!that.parts.empty() && AreAllFullSliceParts(that.parts, 0))) {
      return DesignatorRelation::Equal;
    }
    return parts.empty() ? DesignatorRelation::Contains
                         : DesignatorRelation::ContainedBy;
  }
  bool contains{false};
  bool containedBy{false};
  bool overlaps{false};
  const std::size_t commonSize{
      parts.size() < that.parts.size() ? parts.size() : that.parts.size()};
  for (std::size_t i{0}; i < commonSize; ++i) {
    switch (CompareParts(parts[i], that.parts[i])) {
    case DesignatorRelation::Equal:
      break;
    case DesignatorRelation::Contains:
      contains = true;
      break;
    case DesignatorRelation::ContainedBy:
      containedBy = true;
      break;
    case DesignatorRelation::Overlaps:
      overlaps = true;
      break;
    case DesignatorRelation::Disjoint:
      return DesignatorRelation::Disjoint;
    }
  }
  if (parts.size() < that.parts.size()) {
    if (!AreAllFullSliceParts(that.parts, parts.size())) {
      contains = true;
    }
  } else if (that.parts.size() < parts.size()) {
    if (!AreAllFullSliceParts(parts, that.parts.size())) {
      containedBy = true;
    }
  }
  return CombineRelations(contains, containedBy, overlaps);
}

bool DesignatorPath::MayContain(const DesignatorPath &that) const {
  if (*this == that || empty()) {
    return true;
  }
  if (base || that.base) {
    if (!base || !that.base || !(*base == *that.base)) {
      return false;
    }
  }
  if (that.parts.empty()) {
    return AreAllFullSliceParts(parts, 0);
  }
  if (parts.size() > that.parts.size() &&
      !AreAllFullSliceParts(parts, that.parts.size())) {
    return false;
  }
  if (parts.empty()) {
    return true;
  }
  for (std::size_t i{0}; i < parts.size(); ++i) {
    if (i >= that.parts.size()) {
      return AreAllFullSliceParts(parts, i);
    }
    if (!PartMayContain(parts[i], that.parts[i])) {
      return false;
    }
  }
  return true;
}

void DesignatorPath::SetBase(NamedEntity entity) { base = std::move(entity); }

void DesignatorPath::AddComponent(const Symbol &symbol) {
  if (!parts.empty() && !parts.back().symbol) {
    parts.back().symbol = &symbol;
  } else {
    parts.push_back({{}, &symbol});
  }
}

void DesignatorPath::AddSubscripts(std::vector<Subscript> subscripts) {
  parts.push_back({std::move(subscripts), nullptr});
}

void DesignatorPath::AddComponent(const Component &component) {
  AddDataRef(component.base());
  AddComponent(*component.symbol());
}

void DesignatorPath::AddNamedEntity(const NamedEntity &entity) {
  if (const auto *symbol{entity.UnwrapSymbolRef()}) {
    SetBase(NamedEntity{symbol->get()});
  } else if (const auto *component{entity.UnwrapComponent()}) {
    AddComponent(*component);
  }
}

void DesignatorPath::AddArrayRef(const ArrayRef &arrayRef) {
  AddNamedEntity(arrayRef.base());
  AddSubscripts(arrayRef.subscript());
}

void DesignatorPath::AddCoarrayRef(const CoarrayRef &coarrayRef) {
  AddDataRef(coarrayRef.base());
}

void DesignatorPath::AddDataRef(const DataRef &dataRef) {
  common::visit(
      common::visitors{
          [&](SymbolRef symbol) { SetBase(NamedEntity{symbol.get()}); },
          [&](const Component &component) { AddComponent(component); },
          [&](const ArrayRef &arrayRef) { AddArrayRef(arrayRef); },
          [&](const CoarrayRef &coarrayRef) { AddCoarrayRef(coarrayRef); },
      },
      dataRef.u);
}

std::optional<DesignatorPath> DesignatorPath::Get(
    const std::optional<Expr<SomeType>> &expr) {
  if (std::optional<DataRef> dataRef{ExtractDataRef(expr)}) {
    DesignatorPath path;
    path.AddDataRef(*dataRef);
    if (!path.empty()) {
      return path;
    }
  }
  return std::nullopt;
}

} // namespace Fortran::evaluate
