//===-- lib/Evaluate/type.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/type.h"
#include "flang/Common/idioms.h"
#include "flang/Common/type-kinds.h"
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
  if (IsDescriptor(details.type()) || details.IsAssumedRank()) {
    return true;
  }
  for (const ShapeSpec &shapeSpec : details.shape()) {
    if (const auto &ub{shapeSpec.ubound().GetExplicit()}) {
      if (!IsConstantExpr(*ub)) {
        return true;
      }
    } else {
      return shapeSpec.ubound().isColon();
    }
  }
  return false;
}

bool IsDescriptor(const Symbol &symbol) {
  return common::visit(
      common::visitors{
          [&](const ObjectEntityDetails &d) {
            return IsAllocatableOrPointer(symbol) || IsDescriptor(d);
          },
          [&](const ProcEntityDetails &d) { return false; },
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

bool IsPassedViaDescriptor(const Symbol &symbol) {
  if (!IsDescriptor(symbol)) {
    return false;
  }
  if (IsAllocatableOrPointer(symbol)) {
    return true;
  }
  if (semantics::IsAssumedSizeArray(symbol)) {
    return false;
  }
  if (const auto *object{
          symbol.GetUltimate().detailsIf<ObjectEntityDetails>()}) {
    if (object->isDummy()) {
      if (object->type() &&
          object->type()->category() == DeclTypeSpec::Character) {
        return false;
      }
      bool isExplicitShape{true};
      for (const ShapeSpec &shapeSpec : object->shape()) {
        if (!shapeSpec.lbound().GetExplicit() ||
            !shapeSpec.ubound().GetExplicit()) {
          isExplicitShape = false;
          break;
        }
      }
      if (isExplicitShape) {
        return false; // explicit shape but non-constant bounds
      }
    }
  }
  return true;
}
} // namespace Fortran::semantics

namespace Fortran::evaluate {

DynamicType::DynamicType(int k, const semantics::ParamValue &pv)
    : category_{TypeCategory::Character}, kind_{k} {
  CHECK(common::IsValidKindOfIntrinsicType(category_, kind_));
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
    switch (GetDerivedTypeSpec().category()) {
      SWITCH_COVERS_ALL_CASES
    case semantics::DerivedTypeSpec::Category::DerivedType:
      if (derived_ && derived_->scope()) {
        return derived_->scope()->alignment().value_or(1);
      }
      break;
    case semantics::DerivedTypeSpec::Category::IntrinsicVector:
    case semantics::DerivedTypeSpec::Category::PairVector:
    case semantics::DerivedTypeSpec::Category::QuadVector:
      if (derived_ && derived_->scope()) {
        return derived_->scope()->size();
      } else {
        common::die("Missing scope for Vector type.");
      }
    }
  } else {
    return targetCharacteristics.GetAlignment(category_, kind());
  }
  return 1; // needs to be after switch to dodge a bogus gcc warning
}

std::optional<Expr<SubscriptInteger>> DynamicType::MeasureSizeInBytes(
    FoldingContext &context, bool aligned,
    std::optional<std::int64_t> charLength) const {
  switch (category_) {
  case TypeCategory::Integer:
  case TypeCategory::Unsigned:
  case TypeCategory::Real:
  case TypeCategory::Complex:
  case TypeCategory::Logical:
    return Expr<SubscriptInteger>{
        context.targetCharacteristics().GetByteSize(category_, kind())};
  case TypeCategory::Character:
    if (auto len{charLength ? Expr<SubscriptInteger>{Constant<SubscriptInteger>{
                                  *charLength}}
                            : GetCharLength()}) {
      return Fold(context,
          Expr<SubscriptInteger>{
              context.targetCharacteristics().GetByteSize(category_, kind())} *
              std::move(*len));
    }
    break;
  case TypeCategory::Derived:
    if (!IsPolymorphic() && derived_ && derived_->scope()) {
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

bool DynamicType::IsLengthlessIntrinsicType() const {
  return common::IsNumericTypeCategory(category_) ||
      category_ == TypeCategory::Logical;
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
  const semantics::Scope *scope{derived.scope()};
  if (!scope) {
    scope = typeSymbol.scope();
  }
  if (scope) {
    const auto &dtDetails{typeSymbol.get<semantics::DerivedTypeDetails>()};
    // TODO: Combine with semantics::DerivedTypeDetails::GetParentComponent
    if (auto extends{dtDetails.GetParentComponentName()}) {
      if (auto iter{scope->find(*extends)}; iter != scope->cend()) {
        if (const semantics::Symbol & symbol{*iter->second};
            symbol.test(semantics::Symbol::Flag::ParentComp)) {
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
// represent the "same type" in the sense of section F'2023 7.5.2.4.
using SetOfDerivedTypePairs =
    std::set<std::pair<const semantics::DerivedTypeSpec *,
        const semantics::DerivedTypeSpec *>>;

static bool AreSameDerivedType(const semantics::DerivedTypeSpec &,
    const semantics::DerivedTypeSpec &, bool ignoreTypeParameterValues,
    bool ignoreLenParameters, bool ignoreSequence,
    SetOfDerivedTypePairs &inProgress);

// F2023 7.5.3.2
static bool AreSameComponent(const semantics::Symbol &x,
    const semantics::Symbol &y, bool ignoreSequence, bool sameModuleName,
    SetOfDerivedTypePairs &inProgress) {
  if (x.attrs() != y.attrs()) {
    return false;
  }
  if (x.attrs().test(semantics::Attr::PRIVATE) ||
      y.attrs().test(semantics::Attr::PRIVATE)) {
    if (!sameModuleName ||
        x.attrs().test(semantics::Attr::PRIVATE) !=
            y.attrs().test(semantics::Attr::PRIVATE)) {
      return false;
    }
  }
  if (x.size() && y.size()) {
    if (x.offset() != y.offset() || x.size() != y.size()) {
      return false;
    }
  }
  const auto *xObj{x.detailsIf<semantics::ObjectEntityDetails>()};
  const auto *yObj{y.detailsIf<semantics::ObjectEntityDetails>()};
  const auto *xProc{x.detailsIf<semantics::ProcEntityDetails>()};
  const auto *yProc{y.detailsIf<semantics::ProcEntityDetails>()};
  if (!xObj != !yObj || !xProc != !yProc) {
    return false;
  }
  auto xType{DynamicType::From(x)};
  auto yType{DynamicType::From(y)};
  if (xType && yType) {
    if (xType->category() == TypeCategory::Derived) {
      if (yType->category() != TypeCategory::Derived ||
          !xType->IsUnlimitedPolymorphic() !=
              !yType->IsUnlimitedPolymorphic() ||
          (!xType->IsUnlimitedPolymorphic() &&
              !AreSameDerivedType(xType->GetDerivedTypeSpec(),
                  yType->GetDerivedTypeSpec(), false, false, ignoreSequence,
                  inProgress))) {
        return false;
      }
    } else if (!xType->IsTkLenCompatibleWith(*yType)) {
      return false;
    }
  } else if (xType || yType || !(xProc && yProc)) {
    return false;
  }
  if (xProc) {
    // TODO: compare argument types, &c.
  }
  return true;
}

// TODO: These utilities were cloned out of Semantics to avoid a cyclic
// dependency and should be repackaged into then "namespace semantics"
// part of Evaluate/tools.cpp.

static const semantics::Symbol *GetParentComponent(
    const semantics::DerivedTypeDetails &details,
    const semantics::Scope &scope) {
  if (auto extends{details.GetParentComponentName()}) {
    if (auto iter{scope.find(*extends)}; iter != scope.cend()) {
      if (const Symbol & symbol{*iter->second};
          symbol.test(semantics::Symbol::Flag::ParentComp)) {
        return &symbol;
      }
    }
  }
  return nullptr;
}

static const semantics::Symbol *GetParentComponent(
    const semantics::Symbol *symbol, const semantics::Scope &scope) {
  if (symbol) {
    if (const auto *dtDetails{
            symbol->detailsIf<semantics::DerivedTypeDetails>()}) {
      return GetParentComponent(*dtDetails, scope);
    }
  }
  return nullptr;
}

static const semantics::DerivedTypeSpec *GetParentTypeSpec(
    const semantics::Symbol *symbol, const semantics::Scope &scope) {
  if (const Symbol * parentComponent{GetParentComponent(symbol, scope)}) {
    return &parentComponent->get<semantics::ObjectEntityDetails>()
                .type()
                ->derivedTypeSpec();
  } else {
    return nullptr;
  }
}

static const semantics::Scope *GetDerivedTypeParent(
    const semantics::Scope *scope) {
  if (scope) {
    CHECK(scope->IsDerivedType());
    if (const auto *parent{GetParentTypeSpec(scope->GetSymbol(), *scope)}) {
      return parent->scope();
    }
  }
  return nullptr;
}

static const semantics::Symbol *FindComponent(
    const semantics::Scope *scope, parser::CharBlock name) {
  if (!scope) {
    return nullptr;
  }
  CHECK(scope->IsDerivedType());
  auto found{scope->find(name)};
  if (found != scope->end()) {
    return &*found->second;
  } else {
    return FindComponent(GetDerivedTypeParent(scope), name);
  }
}

static bool AreTypeParamCompatible(const semantics::DerivedTypeSpec &x,
    const semantics::DerivedTypeSpec &y, bool ignoreLenParameters) {
  const auto *xScope{x.typeSymbol().scope()};
  const auto *yScope{y.typeSymbol().scope()};
  for (const auto &[paramName, value] : x.parameters()) {
    const auto *yValue{y.FindParameter(paramName)};
    if (!yValue) {
      return false;
    }
    const auto *xParm{FindComponent(xScope, paramName)};
    const auto *yParm{FindComponent(yScope, paramName)};
    if (xParm && yParm) {
      const auto *xTPD{xParm->detailsIf<semantics::TypeParamDetails>()};
      const auto *yTPD{yParm->detailsIf<semantics::TypeParamDetails>()};
      if (xTPD && yTPD) {
        if (xTPD->attr() != yTPD->attr()) {
          return false;
        }
        if (!ignoreLenParameters ||
            xTPD->attr() != common::TypeParamAttr::Len) {
          auto xExpr{value.GetExplicit()};
          auto yExpr{yValue->GetExplicit()};
          if (xExpr && yExpr) {
            auto xVal{ToInt64(*xExpr)};
            auto yVal{ToInt64(*yExpr)};
            if (xVal && yVal && *xVal != *yVal) {
              return false;
            }
          }
        }
      }
    }
  }
  for (const auto &[paramName, _] : y.parameters()) {
    if (!x.FindParameter(paramName)) {
      return false; // y has more parameters than x
    }
  }
  return true;
}

// F2023 7.5.3.2
static bool AreSameDerivedType(const semantics::DerivedTypeSpec &x,
    const semantics::DerivedTypeSpec &y, bool ignoreTypeParameterValues,
    bool ignoreLenParameters, bool ignoreSequence,
    SetOfDerivedTypePairs &inProgress) {
  if (&x == &y) {
    return true;
  }
  if (!ignoreTypeParameterValues &&
      !AreTypeParamCompatible(x, y, ignoreLenParameters)) {
    return false;
  }
  const auto &xSymbol{x.typeSymbol().GetUltimate()};
  const auto &ySymbol{y.typeSymbol().GetUltimate()};
  if (xSymbol == ySymbol) {
    return true;
  }
  if (xSymbol.name() != ySymbol.name()) {
    return false;
  }
  auto thisQuery{std::make_pair(&x, &y)};
  if (inProgress.find(thisQuery) != inProgress.end()) {
    return true; // recursive use of types in components
  }
  inProgress.insert(thisQuery);
  const auto &xDetails{xSymbol.get<semantics::DerivedTypeDetails>()};
  const auto &yDetails{ySymbol.get<semantics::DerivedTypeDetails>()};
  if (xDetails.sequence() != yDetails.sequence() ||
      xSymbol.attrs().test(semantics::Attr::BIND_C) !=
          ySymbol.attrs().test(semantics::Attr::BIND_C)) {
    return false;
  }
  bool sameModuleName{false};
  const semantics::Scope &xOwner{xSymbol.owner()};
  const semantics::Scope &yOwner{ySymbol.owner()};
  if (xOwner.IsModule() && yOwner.IsModule()) {
    if (auto xModuleName{xOwner.GetName()}) {
      if (auto yModuleName{yOwner.GetName()}) {
        if (*xModuleName == *yModuleName) {
          sameModuleName = true;
        }
      }
    }
  }
  if (!sameModuleName && !ignoreSequence && !xDetails.sequence() &&
      !xSymbol.attrs().test(semantics::Attr::BIND_C)) {
    // PGI does not enforce this requirement; all other Fortran
    // compilers do with a hard error when violations are caught.
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
        yLookup == ySymbol.scope()->end()) {
      return false;
    } else if (!AreSameComponent(*xLookup->second, *yLookup->second,
                   ignoreSequence, sameModuleName, inProgress)) {
      return false;
    }
  }
  return yComponentName == yEnd;
}

bool AreSameDerivedType(
    const semantics::DerivedTypeSpec &x, const semantics::DerivedTypeSpec &y) {
  SetOfDerivedTypePairs inProgress;
  return AreSameDerivedType(x, y, false, false, false, inProgress);
}

bool AreSameDerivedTypeIgnoringTypeParameters(
    const semantics::DerivedTypeSpec &x, const semantics::DerivedTypeSpec &y) {
  SetOfDerivedTypePairs inProgress;
  return AreSameDerivedType(x, y, true, true, false, inProgress);
}

bool AreSameDerivedTypeIgnoringSequence(
    const semantics::DerivedTypeSpec &x, const semantics::DerivedTypeSpec &y) {
  SetOfDerivedTypePairs inProgress;
  return AreSameDerivedType(x, y, false, false, true, inProgress);
}

static bool AreSameDerivedType(
    const semantics::DerivedTypeSpec *x, const semantics::DerivedTypeSpec *y) {
  return x == y || (x && y && AreSameDerivedType(*x, *y));
}

bool DynamicType::IsEquivalentTo(const DynamicType &that) const {
  return category_ == that.category_ && kind_ == that.kind_ &&
      (charLengthParamValue_ == that.charLengthParamValue_ ||
          (charLengthParamValue_ && that.charLengthParamValue_ &&
              charLengthParamValue_->IsEquivalentInInterface(
                  *that.charLengthParamValue_))) &&
      knownLength().has_value() == that.knownLength().has_value() &&
      (!knownLength() || *knownLength() == *that.knownLength()) &&
      AreSameDerivedType(derived_, that.derived_);
}

static bool AreCompatibleDerivedTypes(const semantics::DerivedTypeSpec *x,
    const semantics::DerivedTypeSpec *y, bool isPolymorphic,
    bool ignoreTypeParameterValues, bool ignoreLenTypeParameters) {
  if (!x || !y) {
    return false;
  } else {
    SetOfDerivedTypePairs inProgress;
    if (AreSameDerivedType(*x, *y, ignoreTypeParameterValues,
            ignoreLenTypeParameters, false, inProgress)) {
      return true;
    } else {
      return isPolymorphic &&
          AreCompatibleDerivedTypes(x, GetParentTypeSpec(*y), true,
              ignoreTypeParameterValues, ignoreLenTypeParameters);
    }
  }
}

static bool AreCompatibleTypes(const DynamicType &x, const DynamicType &y,
    bool ignoreTypeParameterValues, bool ignoreLengths) {
  if (x.IsUnlimitedPolymorphic()) {
    return true;
  } else if (y.IsUnlimitedPolymorphic()) {
    return false;
  } else if (x.category() != y.category()) {
    return false;
  } else if (x.category() == TypeCategory::Character) {
    const auto xLen{x.knownLength()};
    const auto yLen{y.knownLength()};
    return x.kind() == y.kind() &&
        (ignoreLengths || !xLen || !yLen || *xLen == *yLen);
  } else if (x.category() == TypeCategory::Derived) {
    const auto *xdt{GetDerivedTypeSpec(x)};
    const auto *ydt{GetDerivedTypeSpec(y)};
    return AreCompatibleDerivedTypes(
        xdt, ydt, x.IsPolymorphic(), ignoreTypeParameterValues, false);
  } else if (x.IsTypelessIntrinsicArgument()) {
    return y.IsTypelessIntrinsicArgument();
  } else {
    return !y.IsTypelessIntrinsicArgument() && x.kind() == y.kind();
  }
}

// See 7.3.2.3 (5) & 15.5.2.4
bool DynamicType::IsTkCompatibleWith(const DynamicType &that) const {
  return AreCompatibleTypes(*this, that, false, true);
}

bool DynamicType::IsTkCompatibleWith(
    const DynamicType &that, common::IgnoreTKRSet ignoreTKR) const {
  if (ignoreTKR.test(common::IgnoreTKR::Type) &&
      (category() == TypeCategory::Derived ||
          that.category() == TypeCategory::Derived ||
          category() != that.category())) {
    return true;
  } else if (ignoreTKR.test(common::IgnoreTKR::Kind) &&
      category() == that.category()) {
    return true;
  } else {
    return AreCompatibleTypes(*this, that, false, true);
  }
}

bool DynamicType::IsTkLenCompatibleWith(const DynamicType &that) const {
  return AreCompatibleTypes(*this, that, false, false);
}

// 16.9.165
std::optional<bool> DynamicType::SameTypeAs(const DynamicType &that) const {
  bool x{AreCompatibleTypes(*this, that, true, true)};
  bool y{AreCompatibleTypes(that, *this, true, true)};
  if (!x && !y) {
    return false;
  } else if (x && y && !IsPolymorphic() && !that.IsPolymorphic()) {
    return true;
  } else {
    return std::nullopt;
  }
}

// 16.9.76
std::optional<bool> DynamicType::ExtendsTypeOf(const DynamicType &that) const {
  if (IsUnlimitedPolymorphic() || that.IsUnlimitedPolymorphic()) {
    return std::nullopt; // unknown
  }
  const auto *thisDts{evaluate::GetDerivedTypeSpec(*this)};
  const auto *thatDts{evaluate::GetDerivedTypeSpec(that)};
  if (!thisDts || !thatDts) {
    return std::nullopt;
  } else if (!AreCompatibleDerivedTypes(thatDts, thisDts, true, true, true)) {
    // Note that I check *thisDts, not its parent, so that EXTENDS_TYPE_OF()
    // is .true. when they are the same type.  This is technically
    // an implementation-defined case in the standard, but every other
    // compiler works this way.
    if (IsPolymorphic() &&
        AreCompatibleDerivedTypes(thisDts, thatDts, true, true, true)) {
      // 'that' is *this or an extension of *this, and so runtime *this
      // could be an extension of 'that'
      return std::nullopt;
    } else {
      return false;
    }
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
      if (common::IsValidKindOfIntrinsicType(category, *kind)) {
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
      return DynamicType{TypeCategory::Integer, std::max(kind(), that.kind())};
    case TypeCategory::Real:
    case TypeCategory::Complex:
      return that;
    default:
      CRASH_NO_CASE;
    }
    break;
  case TypeCategory::Unsigned:
    switch (that.category_) {
    case TypeCategory::Unsigned:
      return DynamicType{TypeCategory::Unsigned, std::max(kind(), that.kind())};
    default:
      CRASH_NO_CASE;
    }
    break;
  case TypeCategory::Real:
    switch (that.category_) {
    case TypeCategory::Integer:
      return *this;
    case TypeCategory::Real:
      return DynamicType{TypeCategory::Real, std::max(kind(), that.kind())};
    case TypeCategory::Complex:
      return DynamicType{TypeCategory::Complex, std::max(kind(), that.kind())};
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
      return DynamicType{TypeCategory::Complex, std::max(kind(), that.kind())};
    default:
      CRASH_NO_CASE;
    }
    break;
  case TypeCategory::Logical:
    switch (that.category_) {
    case TypeCategory::Logical:
      return DynamicType{TypeCategory::Logical, std::max(kind(), that.kind())};
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

int SelectedCharKind(const std::string &s, int defaultKind) { // F'2023 16.9.180
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

std::optional<bool> IsInteroperableIntrinsicType(const DynamicType &type,
    const common::LanguageFeatureControl *features, bool checkCharLength) {
  switch (type.category()) {
  case TypeCategory::Integer:
  case TypeCategory::Unsigned:
    return true;
  case TypeCategory::Real:
  case TypeCategory::Complex:
    return type.kind() >= 4 /* not a short or half float */ || !features ||
        features->IsEnabled(common::LanguageFeature::CUDA);
  case TypeCategory::Logical:
    return type.kind() == 1; // C_BOOL
  case TypeCategory::Character:
    if (type.kind() != 1) { // C_CHAR
      return false;
    } else if (checkCharLength) {
      if (type.knownLength()) {
        return *type.knownLength() == 1;
      } else {
        return std::nullopt;
      }
    } else {
      return true;
    }
  default:
    // Derived types are tested in Semantics/check-declarations.cpp
    return false;
  }
}

bool IsCUDAIntrinsicType(const DynamicType &type) {
  switch (type.category()) {
  case TypeCategory::Integer:
  case TypeCategory::Logical:
    return type.kind() <= 8;
  case TypeCategory::Real:
    return type.kind() >= 2 && type.kind() <= 8;
  case TypeCategory::Complex:
    return type.kind() == 2 || type.kind() == 4 || type.kind() == 8;
  case TypeCategory::Character:
    return type.kind() == 1;
  default:
    // Derived types are tested in Semantics/check-declarations.cpp
    return false;
  }
}

DynamicType DynamicType::DropNonConstantCharacterLength() const {
  if (charLengthParamValue_ && charLengthParamValue_->isExplicit()) {
    if (std::optional<std::int64_t> len{knownLength()}) {
      return DynamicType(kind_, *len);
    } else {
      return DynamicType(category_, kind_);
    }
  }
  return *this;
}

} // namespace Fortran::evaluate
