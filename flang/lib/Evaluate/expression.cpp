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
#include "flang/Parser/message.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <type_traits>

using namespace Fortran::parser::literals;

namespace Fortran::evaluate {

template <int KIND>
std::optional<Expr<SubscriptInteger>>
Expr<Type<TypeCategory::Character, KIND>>::LEN() const {
  using T = std::optional<Expr<SubscriptInteger>>;
  return std::visit(
      common::visitors{
          [](const Constant<Result> &c) -> T {
            return AsExpr(Constant<SubscriptInteger>{c.LEN()});
          },
          [](const ArrayConstructor<Result> &a) -> T { return a.LEN(); },
          [](const Parentheses<Result> &x) { return x.left().LEN(); },
          [](const Convert<Result> &x) {
            return std::visit(
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
    return std::visit(
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
  return std::visit(
      [](const auto &x) {
        if constexpr (common::HasMember<decltype(x), TypelessExpression>) {
          return 0;
        } else {
          return x.Rank();
        }
      },
      derived().u);
}

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

const Expr<SomeType> *StructureConstructor::Find(
    const Symbol &component) const {
  if (auto iter{values_.find(component)}; iter != values_.end()) {
    return &iter->second.value();
  } else {
    return nullptr;
  }
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
  return std::visit(
      [](const auto &kx) { return std::decay_t<decltype(kx)>::Result::kind; },
      u);
}

int Expr<SomeCharacter>::GetKind() const {
  return std::visit(
      [](const auto &kx) { return std::decay_t<decltype(kx)>::Result::kind; },
      u);
}

std::optional<Expr<SubscriptInteger>> Expr<SomeCharacter>::LEN() const {
  return std::visit([](const auto &kx) { return kx.LEN(); }, u);
}

INSTANTIATE_EXPRESSION_TEMPLATES
} // namespace Fortran::evaluate
