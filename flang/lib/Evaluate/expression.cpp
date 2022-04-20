//===-- lib/Evaluate/expression.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/expression.h"
#include "int-power.h"
#include "flang/Common/idioms.h"
#include "flang/Evaluate/common.h"
#include "flang/Evaluate/tools.h"
#include "flang/Evaluate/variable.h"
#include "flang/Parser/char-block.h"
#include "flang/Parser/message.h"
#include "flang/Semantics/scope.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include "flang/Semantics/type.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <type_traits>

using namespace Fortran::parser::literals;

namespace Fortran::evaluate {

template <int KIND>
std::optional<Expr<SubscriptInteger>>
Expr<Type<TypeCategory::Character, KIND>>::LEN() const {
  using T = std::optional<Expr<SubscriptInteger>>;
  return common::visit(
      common::visitors{
          [](const Constant<Result> &c) -> T {
            return AsExpr(Constant<SubscriptInteger>{c.LEN()});
          },
          [](const ArrayConstructor<Result> &a) -> T { return a.LEN(); },
          [](const Parentheses<Result> &x) { return x.left().LEN(); },
          [](const Convert<Result> &x) {
            return common::visit(
                [&](const auto &kx) { return kx.LEN(); }, x.left().u);
          },
          [](const Concat<KIND> &c) -> T {
            if (auto llen{c.left().LEN()}) {
              if (auto rlen{c.right().LEN()}) {
                return *std::move(llen) + *std::move(rlen);
              }
            }
            return std::nullopt;
          },
          [](const Extremum<Result> &c) -> T {
            if (auto llen{c.left().LEN()}) {
              if (auto rlen{c.right().LEN()}) {
                return Expr<SubscriptInteger>{Extremum<SubscriptInteger>{
                    Ordering::Greater, *std::move(llen), *std::move(rlen)}};
              }
            }
            return std::nullopt;
          },
          [](const Designator<Result> &dr) { return dr.LEN(); },
          [](const FunctionRef<Result> &fr) { return fr.LEN(); },
          [](const SetLength<KIND> &x) -> T { return x.right(); },
      },
      u);
}

Expr<SomeType>::~Expr() = default;

#if defined(__APPLE__) && defined(__GNUC__)
template <typename A>
typename ExpressionBase<A>::Derived &ExpressionBase<A>::derived() {
  return *static_cast<Derived *>(this);
}

template <typename A>
const typename ExpressionBase<A>::Derived &ExpressionBase<A>::derived() const {
  return *static_cast<const Derived *>(this);
}
#endif

template <typename A>
std::optional<DynamicType> ExpressionBase<A>::GetType() const {
  if constexpr (IsLengthlessIntrinsicType<Result>) {
    return Result::GetType();
  } else {
    return common::visit(
        [&](const auto &x) -> std::optional<DynamicType> {
          if constexpr (!common::HasMember<decltype(x), TypelessExpression>) {
            return x.GetType();
          }
          return std::nullopt; // w/o "else" to dodge bogus g++ 8.1 warning
        },
        derived().u);
  }
}

template <typename A> int ExpressionBase<A>::Rank() const {
  return common::visit(
      [](const auto &x) {
        if constexpr (common::HasMember<decltype(x), TypelessExpression>) {
          return 0;
        } else {
          return x.Rank();
        }
      },
      derived().u);
}

DynamicType Parentheses<SomeDerived>::GetType() const {
  return left().GetType().value();
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
template <typename A> LLVM_DUMP_METHOD void ExpressionBase<A>::dump() const {
  llvm::errs() << "Expr is <{" << AsFortran() << "}>\n";
}
#endif

// Equality testing

bool ImpliedDoIndex::operator==(const ImpliedDoIndex &that) const {
  return name == that.name;
}

template <typename T>
bool ImpliedDo<T>::operator==(const ImpliedDo<T> &that) const {
  return name_ == that.name_ && lower_ == that.lower_ &&
      upper_ == that.upper_ && stride_ == that.stride_ &&
      values_ == that.values_;
}

template <typename T>
bool ArrayConstructorValue<T>::operator==(
    const ArrayConstructorValue<T> &that) const {
  return u == that.u;
}

template <typename R>
bool ArrayConstructorValues<R>::operator==(
    const ArrayConstructorValues<R> &that) const {
  return values_ == that.values_;
}

template <int KIND>
bool ArrayConstructor<Type<TypeCategory::Character, KIND>>::operator==(
    const ArrayConstructor &that) const {
  return length_ == that.length_ &&
      static_cast<const Base &>(*this) == static_cast<const Base &>(that);
}

bool ArrayConstructor<SomeDerived>::operator==(
    const ArrayConstructor &that) const {
  return result_ == that.result_ &&
      static_cast<const Base &>(*this) == static_cast<const Base &>(that);
  ;
}

StructureConstructor::StructureConstructor(
    const semantics::DerivedTypeSpec &spec,
    const StructureConstructorValues &values)
    : result_{spec}, values_{values} {}
StructureConstructor::StructureConstructor(
    const semantics::DerivedTypeSpec &spec, StructureConstructorValues &&values)
    : result_{spec}, values_{std::move(values)} {}

bool StructureConstructor::operator==(const StructureConstructor &that) const {
  return result_ == that.result_ && values_ == that.values_;
}

bool Relational<SomeType>::operator==(const Relational<SomeType> &that) const {
  return u == that.u;
}

template <int KIND>
bool Expr<Type<TypeCategory::Integer, KIND>>::operator==(
    const Expr<Type<TypeCategory::Integer, KIND>> &that) const {
  return u == that.u;
}

template <int KIND>
bool Expr<Type<TypeCategory::Real, KIND>>::operator==(
    const Expr<Type<TypeCategory::Real, KIND>> &that) const {
  return u == that.u;
}

template <int KIND>
bool Expr<Type<TypeCategory::Complex, KIND>>::operator==(
    const Expr<Type<TypeCategory::Complex, KIND>> &that) const {
  return u == that.u;
}

template <int KIND>
bool Expr<Type<TypeCategory::Logical, KIND>>::operator==(
    const Expr<Type<TypeCategory::Logical, KIND>> &that) const {
  return u == that.u;
}

template <int KIND>
bool Expr<Type<TypeCategory::Character, KIND>>::operator==(
    const Expr<Type<TypeCategory::Character, KIND>> &that) const {
  return u == that.u;
}

template <TypeCategory CAT>
bool Expr<SomeKind<CAT>>::operator==(const Expr<SomeKind<CAT>> &that) const {
  return u == that.u;
}

bool Expr<SomeDerived>::operator==(const Expr<SomeDerived> &that) const {
  return u == that.u;
}

bool Expr<SomeCharacter>::operator==(const Expr<SomeCharacter> &that) const {
  return u == that.u;
}

bool Expr<SomeType>::operator==(const Expr<SomeType> &that) const {
  return u == that.u;
}

DynamicType StructureConstructor::GetType() const { return result_.GetType(); }

std::optional<Expr<SomeType>> StructureConstructor::CreateParentComponent(
    const Symbol &component) const {
  if (const semantics::DerivedTypeSpec *
      parentSpec{GetParentTypeSpec(derivedTypeSpec())}) {
    StructureConstructor structureConstructor{*parentSpec};
    if (const auto *parentDetails{
            component.detailsIf<semantics::DerivedTypeDetails>()}) {
      auto parentIter{parentDetails->componentNames().begin()};
      for (const auto &childIter : values_) {
        if (parentIter == parentDetails->componentNames().end()) {
          break; // There are more components in the child
        }
        SymbolRef componentSymbol{childIter.first};
        structureConstructor.Add(
            *componentSymbol, common::Clone(childIter.second.value()));
        ++parentIter;
      }
      Constant<SomeDerived> constResult{std::move(structureConstructor)};
      Expr<SomeDerived> result{std::move(constResult)};
      return std::optional<Expr<SomeType>>{result};
    }
  }
  return std::nullopt;
}

static const Symbol *GetParentComponentSymbol(const Symbol &symbol) {
  if (symbol.test(Symbol::Flag::ParentComp)) {
    // we have a created parent component
    const auto &compObject{symbol.get<semantics::ObjectEntityDetails>()};
    if (const semantics::DeclTypeSpec * compType{compObject.type()}) {
      const semantics::DerivedTypeSpec &dtSpec{compType->derivedTypeSpec()};
      const semantics::Symbol &compTypeSymbol{dtSpec.typeSymbol()};
      return &compTypeSymbol;
    }
  }
  if (symbol.detailsIf<semantics::DerivedTypeDetails>()) {
    // we have an implicit parent type component
    return &symbol;
  }
  return nullptr;
}

std::optional<Expr<SomeType>> StructureConstructor::Find(
    const Symbol &component) const {
  if (auto iter{values_.find(component)}; iter != values_.end()) {
    return iter->second.value();
  }
  // The component wasn't there directly, see if we're looking for the parent
  // component of an extended type
  if (const Symbol * typeSymbol{GetParentComponentSymbol(component)}) {
    return CreateParentComponent(*typeSymbol);
  }
  // Look for the component in the parent type component.  The parent type
  // component is always the first one
  if (!values_.empty()) {
    const Expr<SomeType> *parentExpr{&values_.begin()->second.value()};
    if (const Expr<SomeDerived> *derivedExpr{
            std::get_if<Expr<SomeDerived>>(&parentExpr->u)}) {
      if (const Constant<SomeDerived> *constExpr{
              std::get_if<Constant<SomeDerived>>(&derivedExpr->u)}) {
        if (std::optional<StructureConstructor> parentComponentValue{
                constExpr->GetScalarValue()}) {
          // Try to find the component in the parent structure constructor
          return parentComponentValue->Find(component);
        }
      }
    }
  }
  return std::nullopt;
}

StructureConstructor &StructureConstructor::Add(
    const Symbol &symbol, Expr<SomeType> &&expr) {
  values_.emplace(symbol, std::move(expr));
  return *this;
}

GenericExprWrapper::~GenericExprWrapper() {}

void GenericExprWrapper::Deleter(GenericExprWrapper *p) { delete p; }

GenericAssignmentWrapper::~GenericAssignmentWrapper() {}

void GenericAssignmentWrapper::Deleter(GenericAssignmentWrapper *p) {
  delete p;
}

template <TypeCategory CAT> int Expr<SomeKind<CAT>>::GetKind() const {
  return common::visit(
      [](const auto &kx) { return std::decay_t<decltype(kx)>::Result::kind; },
      u);
}

int Expr<SomeCharacter>::GetKind() const {
  return common::visit(
      [](const auto &kx) { return std::decay_t<decltype(kx)>::Result::kind; },
      u);
}

std::optional<Expr<SubscriptInteger>> Expr<SomeCharacter>::LEN() const {
  return common::visit([](const auto &kx) { return kx.LEN(); }, u);
}

#ifdef _MSC_VER // disable bogus warning about missing definitions
#pragma warning(disable : 4661)
#endif
INSTANTIATE_EXPRESSION_TEMPLATES
} // namespace Fortran::evaluate
