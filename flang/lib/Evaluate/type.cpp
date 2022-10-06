//===-- lib/Evaluate/type.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/type.h"
#include "flang/Common/idioms.h"
#include "flang/Evaluate/expression.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/target.h"
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
namespace Fortran::semantics {

static bool IsDescriptor(const DeclTypeSpec *type) {
  if (type) {
    if (auto dynamicType{evaluate::DynamicType::From(*type)}) {
      return dynamicType->RequiresDescriptor();
    }
  }
  return false;
}

static bool IsDescriptor(const ObjectEntityDetails &details) {
  if (IsDescriptor(details.type())) {
    return true;
  }
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
  return common::visit(
      common::visitors{
          [&](const ObjectEntityDetails &d) {
            return IsAllocatableOrPointer(symbol) || IsDescriptor(d);
          },
          [&](const ProcEntityDetails &d) {
            return (symbol.attrs().test(Attr::POINTER) ||
                       symbol.attrs().test(Attr::EXTERNAL)) &&
                IsDescriptor(d);
          },
          [&](const EntityDetails &d) { return IsDescriptor(d.type()); },
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

DynamicType::DynamicType(int k, const semantics::ParamValue &pv)
    : category_{TypeCategory::Character}, kind_{k} {
  CHECK(IsValidKindOfIntrinsicType(category_, kind_));
  if (auto n{ToInt64(pv.GetExplicit())}) {
    knownLength_ = *n > 0 ? *n : 0;
  } else {
    charLengthParamValue_ = &pv;
  }
}

template <typename A> inline bool PointeeComparison(const A *x, const A *y) {
  return x == y || (x && y && *x == *y);
}

bool DynamicType::operator==(const DynamicType &that) const {
  return category_ == that.category_ && kind_ == that.kind_ &&
      PointeeComparison(charLengthParamValue_, that.charLengthParamValue_) &&
      knownLength().has_value() == that.knownLength().has_value() &&
      (!knownLength() || *knownLength() == *that.knownLength()) &&
      PointeeComparison(derived_, that.derived_);
}

std::optional<Expr<SubscriptInteger>> DynamicType::GetCharLength() const {
  if (category_ == TypeCategory::Character) {
    if (knownLength()) {
      return AsExpr(Constant<SubscriptInteger>(*knownLength()));
    } else if (charLengthParamValue_) {
      if (auto length{charLengthParamValue_->GetExplicit()}) {
        return ConvertToType<SubscriptInteger>(std::move(*length));
      }
    }
  }
  return std::nullopt;
}

std::size_t DynamicType::GetAlignment(
    const TargetCharacteristics &targetCharacteristics) const {
  if (category_ == TypeCategory::Derived) {
    if (derived_ && derived_->scope()) {
      return derived_->scope()->alignment().value_or(1);
    }
  } else {
    return targetCharacteristics.GetAlignment(category_, kind_);
  }
  return 1; // needs to be after switch to dodge a bogus gcc warning
}

std::optional<Expr<SubscriptInteger>> DynamicType::MeasureSizeInBytes(
    FoldingContext &context, bool aligned) const {
  switch (category_) {
  case TypeCategory::Integer:
  case TypeCategory::Real:
  case TypeCategory::Complex:
  case TypeCategory::Logical:
    return Expr<SubscriptInteger>{
        context.targetCharacteristics().GetByteSize(category_, kind_)};
  case TypeCategory::Character:
    if (auto len{GetCharLength()}) {
      return Fold(context,
          Expr<SubscriptInteger>{
              context.targetCharacteristics().GetByteSize(category_, kind_)} *
              std::move(*len));
    }
    break;
  case TypeCategory::Derived:
    if (derived_ && derived_->scope()) {
      auto size{derived_->scope()->size()};
      auto align{aligned ? derived_->scope()->alignment().value_or(0) : 0};
      auto alignedSize{align > 0 ? ((size + align - 1) / align) * align : size};
      return Expr<SubscriptInteger>{
          static_cast<ConstantSubscript>(alignedSize)};
    }
    break;
  }
  return std::nullopt;
}

bool DynamicType::IsAssumedLengthCharacter() const {
  return category_ == TypeCategory::Character && charLengthParamValue_ &&
      charLengthParamValue_->isAssumed();
}

bool DynamicType::IsNonConstantLengthCharacter() const {
  if (category_ != TypeCategory::Character) {
    return false;
  } else if (knownLength()) {
    return false;
  } else if (!charLengthParamValue_) {
    return true;
  } else if (const auto &expr{charLengthParamValue_->GetExplicit()}) {
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

const semantics::DerivedTypeSpec *GetParentTypeSpec(
    const semantics::DerivedTypeSpec &derived) {
  if (const semantics::Symbol * parent{FindParentComponent(derived)}) {
    return &parent->get<semantics::ObjectEntityDetails>()
                .type()
                ->derivedTypeSpec();
  } else {
    return nullptr;
  }
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
  // TODO: compare types, parameters, bounds, &c.
  return x.has<semantics::ObjectEntityDetails>() ==
      y.has<semantics::ObjectEntityDetails>();
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

static bool AreCompatibleTypes(const DynamicType &x, const DynamicType &y,
    bool ignoreTypeParameterValues) {
  if (x.IsUnlimitedPolymorphic()) {
    return true;
  } else if (y.IsUnlimitedPolymorphic()) {
    return false;
  } else if (x.category() != y.category()) {
    return false;
  } else if (x.category() != TypeCategory::Derived) {
    return x.kind() == y.kind();
  } else {
    const auto *xdt{GetDerivedTypeSpec(x)};
    const auto *ydt{GetDerivedTypeSpec(y)};
    return AreCompatibleDerivedTypes(xdt, ydt, x.IsPolymorphic()) &&
        (ignoreTypeParameterValues ||
            (xdt && ydt && AreTypeParamCompatible(*xdt, *ydt)));
  }
}

// See 7.3.2.3 (5) & 15.5.2.4
bool DynamicType::IsTkCompatibleWith(const DynamicType &that) const {
  return AreCompatibleTypes(*this, that, false);
}

// 16.9.165
std::optional<bool> DynamicType::SameTypeAs(const DynamicType &that) const {
  bool x{AreCompatibleTypes(*this, that, true)};
  bool y{AreCompatibleTypes(that, *this, true)};
  if (x == y) {
    return x;
  } else {
    // If either is unlimited polymorphic, the result is unknown.
    return std::nullopt;
  }
}

// 16.9.76
std::optional<bool> DynamicType::ExtendsTypeOf(const DynamicType &that) const {
  if (IsUnlimitedPolymorphic() || that.IsUnlimitedPolymorphic()) {
    return std::nullopt; // unknown
  } else if (!AreCompatibleDerivedTypes(evaluate::GetDerivedTypeSpec(that),
                 evaluate::GetDerivedTypeSpec(*this), true)) {
    return false;
  } else if (that.IsPolymorphic()) {
    return std::nullopt; // unknown
  } else {
    return true;
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
  return IsPolymorphic() || IsNonConstantLengthCharacter() ||
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
  return charLengthParamValue_ && charLengthParamValue_->isDeferred();
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

std::optional<DynamicType> ComparisonType(
    const DynamicType &t1, const DynamicType &t2) {
  switch (t1.category()) {
  case TypeCategory::Integer:
    switch (t2.category()) {
    case TypeCategory::Integer:
      return DynamicType{TypeCategory::Integer, std::max(t1.kind(), t2.kind())};
    case TypeCategory::Real:
    case TypeCategory::Complex:
      return t2;
    default:
      return std::nullopt;
    }
  case TypeCategory::Real:
    switch (t2.category()) {
    case TypeCategory::Integer:
      return t1;
    case TypeCategory::Real:
    case TypeCategory::Complex:
      return DynamicType{t2.category(), std::max(t1.kind(), t2.kind())};
    default:
      return std::nullopt;
    }
  case TypeCategory::Complex:
    switch (t2.category()) {
    case TypeCategory::Integer:
      return t1;
    case TypeCategory::Real:
    case TypeCategory::Complex:
      return DynamicType{TypeCategory::Complex, std::max(t1.kind(), t2.kind())};
    default:
      return std::nullopt;
    }
  case TypeCategory::Character:
    switch (t2.category()) {
    case TypeCategory::Character:
      return DynamicType{
          TypeCategory::Character, std::max(t1.kind(), t2.kind())};
    default:
      return std::nullopt;
    }
  case TypeCategory::Logical:
    switch (t2.category()) {
    case TypeCategory::Logical:
      return DynamicType{TypeCategory::Logical, LogicalResult::kind};
    default:
      return std::nullopt;
    }
  default:
    return std::nullopt;
  }
}

} // namespace Fortran::evaluate
