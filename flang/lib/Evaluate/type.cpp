//===-- lib/Evaluate/type.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/type.h"
#include "flang/Common/idioms.h"
#include "flang/Common/template.h"
#include "flang/Evaluate/expression.h"
#include "flang/Evaluate/fold.h"
#include "flang/Parser/characters.h"
#include "flang/Semantics/scope.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include "flang/Semantics/type.h"
#include <algorithm>
#include <optional>
#include <string>

// IsDescriptor() predicate: true when a symbol is implemented
// at runtime with a descriptor.
// TODO there's probably a better place for this predicate than here
namespace Fortran::semantics {

static bool IsDescriptor(const ObjectEntityDetails &details) {
  if (const auto *type{details.type()}) {
    if (auto dynamicType{evaluate::DynamicType::From(*type)}) {
      if (dynamicType->RequiresDescriptor()) {
        return true;
      }
    }
  }
  // TODO: Automatic (adjustable) arrays - are they descriptors?
  for (const ShapeSpec &shapeSpec : details.shape()) {
    const auto &lb{shapeSpec.lbound().GetExplicit()};
    const auto &ub{shapeSpec.ubound().GetExplicit()};
    if (!lb || !ub || !IsConstantExpr(*lb) || !IsConstantExpr(*ub)) {
      return true;
    }
  }
  return false;
}

static bool IsDescriptor(const ProcEntityDetails &details) {
  // A procedure pointer or dummy procedure must be & is a descriptor if
  // and only if it requires a static link.
  // TODO: refine this placeholder
  return details.HasExplicitInterface();
}

bool IsDescriptor(const Symbol &symbol) {
  return std::visit(
      common::visitors{
          [&](const ObjectEntityDetails &d) {
            return IsAllocatableOrPointer(symbol) || IsDescriptor(d);
          },
          [&](const ProcEntityDetails &d) {
            return (symbol.attrs().test(Attr::POINTER) ||
                       symbol.attrs().test(Attr::EXTERNAL)) &&
                IsDescriptor(d);
          },
          [](const AssocEntityDetails &d) {
            if (const auto &expr{d.expr()}) {
              if (expr->Rank() > 0) {
                return true;
              }
              if (const auto dynamicType{expr->GetType()}) {
                if (dynamicType->RequiresDescriptor()) {
                  return true;
                }
              }
            }
            return false;
          },
          [](const SubprogramDetails &d) {
            return d.isFunction() && IsDescriptor(d.result());
          },
          [](const UseDetails &d) { return IsDescriptor(d.symbol()); },
          [](const HostAssocDetails &d) { return IsDescriptor(d.symbol()); },
          [](const auto &) { return false; },
      },
      symbol.details());
}
} // namespace Fortran::semantics

namespace Fortran::evaluate {

template <typename A> inline bool PointeeComparison(const A *x, const A *y) {
  return x == y || (x && y && *x == *y);
}

bool DynamicType::operator==(const DynamicType &that) const {
  return category_ == that.category_ && kind_ == that.kind_ &&
      PointeeComparison(charLength_, that.charLength_) &&
      PointeeComparison(derived_, that.derived_);
}

std::optional<common::ConstantSubscript> DynamicType::GetCharLength() const {
  if (category_ == TypeCategory::Character && charLength_ &&
      charLength_->isExplicit()) {
    if (const auto &len{charLength_->GetExplicit()}) {
      return ToInt64(len);
    }
  }
  return std::nullopt;
}

static constexpr int RealKindBytes(int kind) {
  switch (kind) {
  case 3: // non-IEEE 16-bit format (truncated 32-bit)
    return 2;
  case 10: // 80387 80-bit extended precision
  case 12: // possible variant spelling
    return 16;
  default:
    return kind;
  }
}

std::optional<std::size_t> DynamicType::MeasureSizeInBytes() const {
  switch (category_) {
  case TypeCategory::Integer:
    return kind_;
  case TypeCategory::Real:
    return RealKindBytes(kind_);
  case TypeCategory::Complex:
    return 2 * RealKindBytes(kind_);
  case TypeCategory::Character:
    if (auto len{GetCharLength()}) {
      return kind_ * *len;
    }
    break;
  case TypeCategory::Logical:
    return kind_;
  case TypeCategory::Derived:
    if (derived_ && derived_->scope()) {
      return derived_->scope()->size();
    }
    break;
  }
  return std::nullopt;
}

bool DynamicType::IsAssumedLengthCharacter() const {
  return category_ == TypeCategory::Character && charLength_ &&
      charLength_->isAssumed();
}

bool DynamicType::IsUnknownLengthCharacter() const {
  if (category_ != TypeCategory::Character) {
    return false;
  } else if (!charLength_) {
    return true;
  } else if (const auto &expr{charLength_->GetExplicit()}) {
    return !IsConstantExpr(*expr);
  } else {
    return true;
  }
}

bool DynamicType::IsTypelessIntrinsicArgument() const {
  return category_ == TypeCategory::Integer && kind_ == TypelessKind;
}

const semantics::DerivedTypeSpec *GetDerivedTypeSpec(
    const std::optional<DynamicType> &type) {
  return type ? GetDerivedTypeSpec(*type) : nullptr;
}

const semantics::DerivedTypeSpec *GetDerivedTypeSpec(const DynamicType &type) {
  if (type.category() == TypeCategory::Derived &&
      !type.IsUnlimitedPolymorphic()) {
    return &type.GetDerivedTypeSpec();
  } else {
    return nullptr;
  }
}

static const semantics::Symbol *FindParentComponent(
    const semantics::DerivedTypeSpec &derived) {
  const semantics::Symbol &typeSymbol{derived.typeSymbol()};
  if (const semantics::Scope * scope{typeSymbol.scope()}) {
    const auto &dtDetails{typeSymbol.get<semantics::DerivedTypeDetails>()};
    if (auto extends{dtDetails.GetParentComponentName()}) {
      if (auto iter{scope->find(*extends)}; iter != scope->cend()) {
        if (const Symbol & symbol{*iter->second};
            symbol.test(Symbol::Flag::ParentComp)) {
          return &symbol;
        }
      }
    }
  }
  return nullptr;
}

static const semantics::DerivedTypeSpec *GetParentTypeSpec(
    const semantics::DerivedTypeSpec &derived) {
  if (const semantics::Symbol * parent{FindParentComponent(derived)}) {
    return &parent->get<semantics::ObjectEntityDetails>()
                .type()
                ->derivedTypeSpec();
  } else {
    return nullptr;
  }
}

static const semantics::Symbol *FindComponent(
    const semantics::DerivedTypeSpec &derived, parser::CharBlock name) {
  if (const auto *scope{derived.scope()}) {
    auto iter{scope->find(name)};
    if (iter != scope->end()) {
      return &*iter->second;
    } else if (const auto *parent{GetParentTypeSpec(derived)}) {
      return FindComponent(*parent, name);
    }
  }
  return nullptr;
}

// Compares two derived type representations to see whether they both
// represent the "same type" in the sense of section 7.5.2.4.
using SetOfDerivedTypePairs =
    std::set<std::pair<const semantics::DerivedTypeSpec *,
        const semantics::DerivedTypeSpec *>>;

static bool AreSameComponent(const semantics::Symbol &,
    const semantics::Symbol &, SetOfDerivedTypePairs &inProgress);

static bool AreSameDerivedType(const semantics::DerivedTypeSpec &x,
    const semantics::DerivedTypeSpec &y, SetOfDerivedTypePairs &inProgress) {
  const auto &xSymbol{x.typeSymbol()};
  const auto &ySymbol{y.typeSymbol()};
  if (&x == &y || xSymbol == ySymbol) {
    return true;
  }
  auto thisQuery{std::make_pair(&x, &y)};
  if (inProgress.find(thisQuery) != inProgress.end()) {
    return true; // recursive use of types in components
  }
  inProgress.insert(thisQuery);
  const auto &xDetails{xSymbol.get<semantics::DerivedTypeDetails>()};
  const auto &yDetails{ySymbol.get<semantics::DerivedTypeDetails>()};
  if (xSymbol.name() != ySymbol.name()) {
    return false;
  }
  if (!(xDetails.sequence() && yDetails.sequence()) &&
      !(xSymbol.attrs().test(semantics::Attr::BIND_C) &&
          ySymbol.attrs().test(semantics::Attr::BIND_C))) {
    // PGI does not enforce this requirement; all other Fortran
    // processors do with a hard error when violations are caught.
    return false;
  }
  // Compare the component lists in their orders of declaration.
  auto xEnd{xDetails.componentNames().cend()};
  auto yComponentName{yDetails.componentNames().cbegin()};
  auto yEnd{yDetails.componentNames().cend()};
  for (auto xComponentName{xDetails.componentNames().cbegin()};
       xComponentName != xEnd; ++xComponentName, ++yComponentName) {
    if (yComponentName == yEnd || *xComponentName != *yComponentName ||
        !xSymbol.scope() || !ySymbol.scope()) {
      return false;
    }
    const auto xLookup{xSymbol.scope()->find(*xComponentName)};
    const auto yLookup{ySymbol.scope()->find(*yComponentName)};
    if (xLookup == xSymbol.scope()->end() ||
        yLookup == ySymbol.scope()->end() ||
        !AreSameComponent(*xLookup->second, *yLookup->second, inProgress)) {
      return false;
    }
  }
  return yComponentName == yEnd;
}

static bool AreSameComponent(const semantics::Symbol &x,
    const semantics::Symbol &y,
    SetOfDerivedTypePairs & /* inProgress - not yet used */) {
  if (x.attrs() != y.attrs()) {
    return false;
  }
  if (x.attrs().test(semantics::Attr::PRIVATE)) {
    return false;
  }
#if 0 // TODO
  if (const auto *xObject{x.detailsIf<semantics::ObjectEntityDetails>()}) {
    if (const auto *yObject{y.detailsIf<semantics::ObjectEntityDetails>()}) {
#else
  if (x.has<semantics::ObjectEntityDetails>()) {
    if (y.has<semantics::ObjectEntityDetails>()) {
#endif
  // TODO: compare types, type parameters, bounds, &c.
  return true;
}
else {
  return false;
}
} // namespace Fortran::evaluate
else {
  // TODO: non-object components
  return true;
}
}

static bool AreCompatibleDerivedTypes(const semantics::DerivedTypeSpec *x,
    const semantics::DerivedTypeSpec *y, bool isPolymorphic) {
  if (!x || !y) {
    return false;
  } else {
    SetOfDerivedTypePairs inProgress;
    if (AreSameDerivedType(*x, *y, inProgress)) {
      return true;
    } else {
      return isPolymorphic &&
          AreCompatibleDerivedTypes(x, GetParentTypeSpec(*y), true);
    }
  }
}

bool IsKindTypeParameter(const semantics::Symbol &symbol) {
  const auto *param{symbol.detailsIf<semantics::TypeParamDetails>()};
  return param && param->attr() == common::TypeParamAttr::Kind;
}

static bool IsKindTypeParameter(
    const semantics::DerivedTypeSpec &derived, parser::CharBlock name) {
  const semantics::Symbol *symbol{FindComponent(derived, name)};
  return symbol && IsKindTypeParameter(*symbol);
}

bool DynamicType::IsTypeCompatibleWith(const DynamicType &that) const {
  if (derived_) {
    if (!AreCompatibleDerivedTypes(derived_, that.derived_, IsPolymorphic())) {
      return false;
    }
    // The values of derived type KIND parameters must match.
    for (const auto &[name, param] : derived_->parameters()) {
      if (IsKindTypeParameter(*derived_, name)) {
        bool ok{false};
        if (auto myValue{ToInt64(param.GetExplicit())}) {
          if (const auto *thatParam{that.derived_->FindParameter(name)}) {
            if (auto thatValue{ToInt64(thatParam->GetExplicit())}) {
              ok = *myValue == *thatValue;
            }
          }
        }
        if (!ok) {
          return false;
        }
      }
    }
    return true;
  } else if (category_ == that.category_ && kind_ == that.kind_) {
    // CHARACTER length is not checked here
    return true;
  } else {
    return IsUnlimitedPolymorphic();
  }
}

// Do the kind type parameters of type1 have the same values as the
// corresponding kind type parameters of the type2?
static bool IsKindCompatible(const semantics::DerivedTypeSpec &type1,
    const semantics::DerivedTypeSpec &type2) {
  for (const auto &[name, param1] : type1.parameters()) {
    if (param1.isKind()) {
      const semantics::ParamValue *param2{type2.FindParameter(name)};
      if (!PointeeComparison(&param1, param2)) {
        return false;
      }
    }
  }
  return true;
}

bool DynamicType::IsTkCompatibleWith(const DynamicType &that) const {
  if (category_ != TypeCategory::Derived) {
    return category_ == that.category_ && kind_ == that.kind_;
  } else if (IsUnlimitedPolymorphic()) {
    return true;
  } else if (that.IsUnlimitedPolymorphic()) {
    return false;
  } else if (!derived_ || !that.derived_ ||
      !IsKindCompatible(*derived_, *that.derived_)) {
    return false; // kind params don't match
  } else {
    return AreCompatibleDerivedTypes(derived_, that.derived_, IsPolymorphic());
  }
}

std::optional<DynamicType> DynamicType::From(
    const semantics::DeclTypeSpec &type) {
  if (const auto *intrinsic{type.AsIntrinsic()}) {
    if (auto kind{ToInt64(intrinsic->kind())}) {
      TypeCategory category{intrinsic->category()};
      if (IsValidKindOfIntrinsicType(category, *kind)) {
        if (category == TypeCategory::Character) {
          const auto &charType{type.characterTypeSpec()};
          return DynamicType{static_cast<int>(*kind), charType.length()};
        } else {
          return DynamicType{category, static_cast<int>(*kind)};
        }
      }
    }
  } else if (const auto *derived{type.AsDerived()}) {
    return DynamicType{
        *derived, type.category() == semantics::DeclTypeSpec::ClassDerived};
  } else if (type.category() == semantics::DeclTypeSpec::ClassStar) {
    return DynamicType::UnlimitedPolymorphic();
  } else if (type.category() == semantics::DeclTypeSpec::TypeStar) {
    return DynamicType::AssumedType();
  } else {
    common::die("DynamicType::From(DeclTypeSpec): failed");
  }
  return std::nullopt;
}

std::optional<DynamicType> DynamicType::From(const semantics::Symbol &symbol) {
  return From(symbol.GetType()); // Symbol -> DeclTypeSpec -> DynamicType
}

DynamicType DynamicType::ResultTypeForMultiply(const DynamicType &that) const {
  switch (category_) {
  case TypeCategory::Integer:
    switch (that.category_) {
    case TypeCategory::Integer:
      return DynamicType{TypeCategory::Integer, std::max(kind_, that.kind_)};
    case TypeCategory::Real:
    case TypeCategory::Complex:
      return that;
    default:
      CRASH_NO_CASE;
    }
    break;
  case TypeCategory::Real:
    switch (that.category_) {
    case TypeCategory::Integer:
      return *this;
    case TypeCategory::Real:
      return DynamicType{TypeCategory::Real, std::max(kind_, that.kind_)};
    case TypeCategory::Complex:
      return DynamicType{TypeCategory::Complex, std::max(kind_, that.kind_)};
    default:
      CRASH_NO_CASE;
    }
    break;
  case TypeCategory::Complex:
    switch (that.category_) {
    case TypeCategory::Integer:
      return *this;
    case TypeCategory::Real:
    case TypeCategory::Complex:
      return DynamicType{TypeCategory::Complex, std::max(kind_, that.kind_)};
    default:
      CRASH_NO_CASE;
    }
    break;
  case TypeCategory::Logical:
    switch (that.category_) {
    case TypeCategory::Logical:
      return DynamicType{TypeCategory::Logical, std::max(kind_, that.kind_)};
    default:
      CRASH_NO_CASE;
    }
    break;
  default:
    CRASH_NO_CASE;
  }
  return *this;
}

bool DynamicType::RequiresDescriptor() const {
  return IsPolymorphic() || IsUnknownLengthCharacter() ||
      (derived_ && CountNonConstantLenParameters(*derived_) > 0);
}

bool DynamicType::HasDeferredTypeParameter() const {
  if (derived_) {
    for (const auto &pair : derived_->parameters()) {
      if (pair.second.isDeferred()) {
        return true;
      }
    }
  }
  return charLength_ && charLength_->isDeferred();
}

bool SomeKind<TypeCategory::Derived>::operator==(
    const SomeKind<TypeCategory::Derived> &that) const {
  return PointeeComparison(derivedTypeSpec_, that.derivedTypeSpec_);
}

int SelectedCharKind(const std::string &s, int defaultKind) { // 16.9.168
  auto lower{parser::ToLowerCaseLetters(s)};
  auto n{lower.size()};
  while (n > 0 && lower[0] == ' ') {
    lower.erase(0, 1);
    --n;
  }
  while (n > 0 && lower[n - 1] == ' ') {
    lower.erase(--n, 1);
  }
  if (lower == "ascii") {
    return 1;
  } else if (lower == "ucs-2") {
    return 2;
  } else if (lower == "iso_10646" || lower == "ucs-4") {
    return 4;
  } else if (lower == "default") {
    return defaultKind;
  } else {
    return -1;
  }
}

class SelectedIntKindVisitor {
public:
  explicit SelectedIntKindVisitor(std::int64_t p) : precision_{p} {}
  using Result = std::optional<int>;
  using Types = IntegerTypes;
  template <typename T> Result Test() const {
    if (Scalar<T>::RANGE >= precision_) {
      return T::kind;
    } else {
      return std::nullopt;
    }
  }

private:
  std::int64_t precision_;
};

int SelectedIntKind(std::int64_t precision) {
  if (auto kind{common::SearchTypes(SelectedIntKindVisitor{precision})}) {
    return *kind;
  } else {
    return -1;
  }
}

class SelectedRealKindVisitor {
public:
  explicit SelectedRealKindVisitor(std::int64_t p, std::int64_t r)
      : precision_{p}, range_{r} {}
  using Result = std::optional<int>;
  using Types = RealTypes;
  template <typename T> Result Test() const {
    if (Scalar<T>::PRECISION >= precision_ && Scalar<T>::RANGE >= range_) {
      return {T::kind};
    } else {
      return std::nullopt;
    }
  }

private:
  std::int64_t precision_, range_;
};

int SelectedRealKind(
    std::int64_t precision, std::int64_t range, std::int64_t radix) {
  if (radix != 2) {
    return -5;
  }
  if (auto kind{
          common::SearchTypes(SelectedRealKindVisitor{precision, range})}) {
    return *kind;
  }
  // No kind has both sufficient precision and sufficient range.
  // The negative return value encodes whether any kinds exist that
  // could satisfy either constraint independently.
  bool pOK{common::SearchTypes(SelectedRealKindVisitor{precision, 0})};
  bool rOK{common::SearchTypes(SelectedRealKindVisitor{0, range})};
  if (pOK) {
    if (rOK) {
      return -4;
    } else {
      return -2;
    }
  } else {
    if (rOK) {
      return -1;
    } else {
      return -3;
    }
  }
}
} // namespace Fortran::evaluate
