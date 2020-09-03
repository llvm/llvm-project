//===-- include/flang/Evaluate/fold.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_FOLD_H_
#define FORTRAN_EVALUATE_FOLD_H_

// Implements expression tree rewriting, particularly constant expression
// and designator reference evaluation.

#include "common.h"
#include "constant.h"
#include "expression.h"
#include "tools.h"
#include "type.h"
#include <variant>

namespace Fortran::evaluate {

using namespace Fortran::parser::literals;

// Fold() rewrites an expression and returns it.  When the rewritten expression
// is a constant, UnwrapConstantValue() and GetScalarConstantValue() below will
// be able to extract it.
// Note the rvalue reference argument: the rewrites are performed in place
// for efficiency.
template <typename T> Expr<T> Fold(FoldingContext &context, Expr<T> &&expr) {
  return Expr<T>::Rewrite(context, std::move(expr));
}

template <typename T>
std::optional<Expr<T>> Fold(
    FoldingContext &context, std::optional<Expr<T>> &&expr) {
  if (expr) {
    return Fold(context, std::move(*expr));
  } else {
    return std::nullopt;
  }
}

// UnwrapConstantValue() isolates the known constant value of
// an expression, if it has one.  It returns a pointer, which is
// const-qualified when the expression is so.  The value can be
// parenthesized.
template <typename T, typename EXPR>
auto UnwrapConstantValue(EXPR &expr) -> common::Constify<Constant<T>, EXPR> * {
  if (auto *c{UnwrapExpr<Constant<T>>(expr)}) {
    return c;
  } else {
    if constexpr (!std::is_same_v<T, SomeDerived>) {
      if (auto *parens{UnwrapExpr<Parentheses<T>>(expr)}) {
        return UnwrapConstantValue<T>(parens->left());
      }
    }
    return nullptr;
  }
}

// GetScalarConstantValue() extracts the known scalar constant value of
// an expression, if it has one.  The value can be parenthesized.
template <typename T, typename EXPR>
auto GetScalarConstantValue(const EXPR &expr) -> std::optional<Scalar<T>> {
  if (const Constant<T> *constant{UnwrapConstantValue<T>(expr)}) {
    return constant->GetScalarValue();
  } else {
    return std::nullopt;
  }
}

// When an expression is a constant integer, ToInt64() extracts its value.
// Ensure that the expression has been folded beforehand when folding might
// be required.
template <int KIND>
std::optional<std::int64_t> ToInt64(
    const Expr<Type<TypeCategory::Integer, KIND>> &expr) {
  if (auto scalar{
          GetScalarConstantValue<Type<TypeCategory::Integer, KIND>>(expr)}) {
    return scalar->ToInt64();
  } else {
    return std::nullopt;
  }
}

std::optional<std::int64_t> ToInt64(const Expr<SomeInteger> &);
std::optional<std::int64_t> ToInt64(const Expr<SomeType> &);

template <typename A>
std::optional<std::int64_t> ToInt64(const std::optional<A> &x) {
  if (x) {
    return ToInt64(*x);
  } else {
    return std::nullopt;
  }
}
} // namespace Fortran::evaluate
#endif // FORTRAN_EVALUATE_FOLD_H_
