//===-- lib/Evaluate/designator-path.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/designator-path.h"
#include "flang/Common/idioms.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/tools.h"
#include "flang/Semantics/symbol.h"
#include <cstdint>

namespace Fortran::evaluate {

namespace {
// The values selected by a subscript with constant bounds and stride, as the
// ascending arithmetic progression first, first + stride, ..., last.
struct ConstantSection {
  std::int64_t first;
  std::int64_t last;
  std::int64_t stride;
};

struct ConstantSubscriptRange {
  std::int64_t lower;
  std::int64_t upper;
};
} // namespace

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

static bool IsFullTriplet(const Triplet &triplet) {
  // Surface syntax `x(:)` maps to a Triplet with no lower or upper bound and
  // an implicit stride of one.
  auto stride{ToInt64(triplet.GetStride())};
  return !triplet.GetLower() && !triplet.GetUpper() && stride && *stride == 1;
}

static bool IsFullSubscriptList(const std::vector<Subscript> &subscripts) {
  if (subscripts.empty()) {
    return false;
  }
  for (const Subscript &subscript : subscripts) {
    const auto *triplet{std::get_if<Triplet>(&subscript.u)};
    if (!triplet || !IsFullTriplet(*triplet)) {
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

static std::optional<ConstantSubscriptRange> GetConstantSubscriptRange(
    const Subscript &subscript) {
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

// Like GetConstantSubscriptRange, but accepts any nonzero constant stride.
// Returns nothing for a selection with a nonconstant or missing bound or
// stride, and for an empty section.
static std::optional<ConstantSection> GetConstantSection(
    const Subscript &subscript) {
  if (const auto *expr{
          std::get_if<IndirectSubscriptIntegerExpr>(&subscript.u)}) {
    if (auto value{ToInt64(expr->value())}) {
      return ConstantSection{*value, *value, 1};
    }
    return std::nullopt;
  }
  const auto *triplet{std::get_if<Triplet>(&subscript.u)};
  if (!triplet || !triplet->GetLower() || !triplet->GetUpper()) {
    return std::nullopt;
  }
  auto lower{ToInt64(*triplet->GetLower())};
  auto upper{ToInt64(*triplet->GetUpper())};
  auto stride{ToInt64(triplet->GetStride())};
  if (!lower || !upper || !stride || *stride == 0) {
    return std::nullopt;
  }
  if (*stride > 0) {
    if (*lower > *upper) {
      return std::nullopt;
    }
    std::int64_t count{(*upper - *lower) / *stride};
    return ConstantSection{*lower, *lower + count * *stride, *stride};
  }
  if (*lower < *upper) {
    return std::nullopt;
  }
  std::int64_t count{(*lower - *upper) / -*stride};
  return ConstantSection{*lower + count * *stride, *lower, -*stride};
}

// Every value of y is a value of x. The values of y are an arithmetic
// progression, so they lie in x exactly when the ends do and, if y has more
// than one value, the step of y is a multiple of the step of x.
static bool SectionContains(
    const ConstantSection &x, const ConstantSection &y) {
  auto inX{[&](std::int64_t value) {
    return value >= x.first && value <= x.last &&
        (value - x.first) % x.stride == 0;
  }};
  return inX(y.first) && inX(y.last) &&
      (y.first == y.last || y.stride % x.stride == 0);
}

static DesignatorRelation CompareSubscripts(
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
    // Nonconstant selectors and constant triplets with non-unit strides cannot
    // be compared precisely, so report them as Disjoint for now. Callers must
    // not read that as proof that the selections do not overlap. Constant
    // triplets could be made more precise by expanding them into index sets.
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

static bool SubscriptMayContain(const Subscript &x, const Subscript &y) {
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
  // Decide exactly whenever both selections are constant, including a scalar
  // against a one-element triplet and any constant stride.
  auto xSection{GetConstantSection(x)};
  auto ySection{GetConstantSection(y)};
  if (xSection && ySection) {
    return SectionContains(*xSection, *ySection);
  }
  if (!xTriplet && yTriplet) {
    return false;
  }
  if (xTriplet) {
    return true;
  }
  return !ToInt64(std::get<IndirectSubscriptIntegerExpr>(x.u).value()) ||
      !ToInt64(std::get<IndirectSubscriptIntegerExpr>(y.u).value());
}

static bool SubscriptListMayContain(
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

static bool PartMayContain(
    const DesignatorPath::Part &x, const DesignatorPath::Part &y) {
  return SubscriptListMayContain(x.subscripts, y.subscripts) &&
      (!x.symbol || x.symbol == y.symbol);
}

static DesignatorRelation CombineRelations(
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

// Folds the relation of one subscript or part into the running summary of a
// comparison. Returns false for Disjoint, which decides the whole comparison.
static bool Accumulate(DesignatorRelation relation, bool &contains,
    bool &containedBy, bool &overlaps) {
  switch (relation) {
    SWITCH_COVERS_ALL_CASES
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
    return false;
  }
  return true;
}

static DesignatorRelation CompareSubscriptLists(
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
    if (!Accumulate(
            CompareSubscripts(x[i], y[i]), contains, containedBy, overlaps)) {
      return DesignatorRelation::Disjoint;
    }
  }
  return CombineRelations(contains, containedBy, overlaps);
}

static DesignatorRelation CompareParts(
    const DesignatorPath::Part &x, const DesignatorPath::Part &y) {
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
  if (commonBlock_ && commonBlock_ == that.commonBlock_) {
    if (IsWholeCommonBlock()) {
      return that.IsWholeCommonBlock() ? DesignatorRelation::Equal
                                       : DesignatorRelation::Contains;
    }
    if (that.IsWholeCommonBlock()) {
      return DesignatorRelation::ContainedBy;
    }
  }
  if (base_ || that.base_) {
    if (!base_ || !that.base_ || !(*base_ == *that.base_)) {
      return DesignatorRelation::Disjoint;
    }
  }
  if (parts_.empty() || that.parts_.empty()) {
    if ((!parts_.empty() && AreAllFullSliceParts(parts_, 0)) ||
        (!that.parts_.empty() && AreAllFullSliceParts(that.parts_, 0))) {
      return DesignatorRelation::Equal;
    }
    return parts_.empty() ? DesignatorRelation::Contains
                          : DesignatorRelation::ContainedBy;
  }
  bool contains{false};
  bool containedBy{false};
  bool overlaps{false};
  const std::size_t commonSize{
      parts_.size() < that.parts_.size() ? parts_.size() : that.parts_.size()};
  for (std::size_t i{0}; i < commonSize; ++i) {
    if (!Accumulate(CompareParts(parts_[i], that.parts_[i]), contains,
            containedBy, overlaps)) {
      return DesignatorRelation::Disjoint;
    }
  }
  if (parts_.size() < that.parts_.size()) {
    if (!AreAllFullSliceParts(that.parts_, parts_.size())) {
      contains = true;
    }
  } else if (that.parts_.size() < parts_.size()) {
    if (!AreAllFullSliceParts(parts_, that.parts_.size())) {
      containedBy = true;
    }
  }
  return CombineRelations(contains, containedBy, overlaps);
}

bool DesignatorPath::MayContain(const DesignatorPath &that) const {
  if (*this == that || empty()) {
    return true;
  }
  if (commonBlock_ && commonBlock_ == that.commonBlock_ &&
      IsWholeCommonBlock()) {
    return true;
  }
  if (base_ || that.base_) {
    if (!base_ || !that.base_ || !(*base_ == *that.base_)) {
      return false;
    }
  }
  if (that.parts_.empty()) {
    return AreAllFullSliceParts(parts_, 0);
  }
  if (parts_.size() > that.parts_.size() &&
      !AreAllFullSliceParts(parts_, that.parts_.size())) {
    return false;
  }
  if (parts_.empty()) {
    return true;
  }
  for (std::size_t i{0}; i < parts_.size(); ++i) {
    if (i >= that.parts_.size()) {
      return AreAllFullSliceParts(parts_, i);
    }
    if (!PartMayContain(parts_[i], that.parts_[i])) {
      return false;
    }
  }
  return true;
}

std::string DesignatorPath::AsFortran() const {
  std::string result;
  llvm::raw_string_ostream stream{result};
  AsFortran(stream);
  return result;
}

llvm::raw_ostream &DesignatorPath::AsFortran(llvm::raw_ostream &o) const {
  if (!base_) {
    return o;
  }
  base_->AsFortran(o);
  for (const Part &part : parts_) {
    if (!part.subscripts.empty()) {
      char separator{'('};
      for (const Subscript &subscript : part.subscripts) {
        subscript.AsFortran(o << separator);
        separator = ',';
      }
      o << ')';
    }
    if (part.symbol) {
      o << '%' << part.symbol->name();
    }
  }
  return o;
}

bool DesignatorPath::IsWholeCommonBlock() const {
  return commonBlock_ && HasBaseOnly() && base_->IsSymbol() &&
      &base_->GetFirstSymbol().GetUltimate() == commonBlock_;
}

void DesignatorPath::SetBase(NamedEntity entity) {
  base_ = std::move(entity);
  commonBlock_ = nullptr;
  const Symbol &symbol{base_->GetFirstSymbol().GetUltimate()};
  if (symbol.has<semantics::CommonBlockDetails>()) {
    commonBlock_ = &symbol;
  } else if (const auto *details{
                 symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
    if (const Symbol *block{details->commonBlock()}) {
      commonBlock_ = &block->GetUltimate();
    }
  }
}

void DesignatorPath::AddComponent(const Symbol &symbol) {
  if (!parts_.empty() && !parts_.back().symbol) {
    parts_.back().symbol = &symbol;
  } else {
    parts_.push_back({{}, &symbol});
  }
}

void DesignatorPath::AddSubscripts(std::vector<Subscript> subscripts) {
  parts_.push_back({std::move(subscripts), nullptr});
}

} // namespace Fortran::evaluate
