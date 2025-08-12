//===-- include/flang/Evaluate/rewrite.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef FORTRAN_EVALUATE_REWRITE_H_
#define FORTRAN_EVALUATE_REWRITE_H_

#include "flang/Common/visit.h"
#include "flang/Evaluate/expression.h"
#include "flang/Support/Fortran.h"
#include "llvm/ADT/STLExtras.h"

#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

namespace Fortran::evaluate {
namespace rewrite {
namespace detail {
template <typename, typename = void> //
struct IsOperation {
  static constexpr bool value{false};
};

template <typename T>
struct IsOperation<T, std::void_t<decltype(T::operands)>> {
  static constexpr bool value{true};
};
} // namespace detail

template <typename T>
constexpr bool is_operation_v{detail::IsOperation<T>::value};

/// Individual Expr<T> rewriter that simply constructs an expression that is
/// identical to the input. This is a suitable base class for all user-defined
/// rewriters.
struct Identity {
  template <typename T, typename U>
  Expr<T> operator()(Expr<T> &&x, const U &op) {
    return std::move(x);
  }
};

/// Bottom-up Expr<T> rewriter.
///
/// The Mutator traverses and reconstructs given Expr<T>. Going bottom-up,
/// whenever the traversal visits a sub-node of type Expr<U> (for some U),
/// it will invoke the user-provided rewriter via the () operator.
///
/// If x is of type Expr<U>, it will call (in pseudo-code):
///   rewriter_(x, active_member_of(x.u))
/// The second parameter is there to make it easier to overload the () operator
/// for specific operations in Expr<...>.
///
/// The user rewriter is only invoked for Expr<U>, not for Operation, nor any
/// other subobject.
template <typename Rewriter> struct Mutator {
  Mutator(Rewriter &rewriter) : rewriter_(rewriter) {}

  template <typename T, typename U = llvm::remove_cvref_t<T>>
  U operator()(T &&x) {
    if constexpr (std::is_lvalue_reference_v<T>) {
      return Mutate(U(x));
    } else {
      return Mutate(std::move(x));
    }
  }

private:
  template <typename T> struct LambdaWithRvalueCapture {
    LambdaWithRvalueCapture(Rewriter &r, Expr<T> &&c)
        : rewriter_(r), capture_(std::move(c)) {}
    template <typename S> Expr<T> operator()(const S &s) {
      return rewriter_(std::move(capture_), s);
    }

  private:
    Rewriter &rewriter_;
    Expr<T> &&capture_;
  };

  template <typename T, typename = std::enable_if_t<!is_operation_v<T>>>
  T Mutate(T &&x) const {
    return std::move(x);
  }

  template <typename D, typename = std::enable_if_t<is_operation_v<D>>>
  D Mutate(D &&op, std::make_index_sequence<D::operands> t = {}) const {
    return MutateOp(std::move(op), t);
  }

  template <typename T> //
  Expr<T> Mutate(Expr<T> &&x) const {
    // First construct the new expression with the rewritten op.
    Expr<T> n{common::visit(
        [&](auto &&s) { //
          return Expr<T>(Mutate(std::move(s)));
        },
        std::move(x.u))};
    // Return the rewritten expression. The second visit is to make sure
    // that the second argument in the call to the rewriter is a part of
    // the Expr<T> passed to it.
    return common::visit(
        LambdaWithRvalueCapture<T>(rewriter_, std::move(n)), std::move(n.u));
  }

  template <typename... Ts>
  std::variant<Ts...> Mutate(std::variant<Ts...> &&u) const {
    return common::visit(
        [this](auto &&s) { return Mutate(std::move(s)); }, std::move(u));
  }

  template <typename... Ts>
  std::tuple<Ts...> Mutate(std::tuple<Ts...> &&t) const {
    return MutateTuple(std::move(t), std::index_sequence_for<Ts...>{});
  }

  template <typename... Ts, size_t... Is>
  std::tuple<Ts...> MutateTuple(
      std::tuple<Ts...> &&t, std::index_sequence<Is...>) const {
    return std::make_tuple(Mutate(std::move(std::get<Is>(t))...));
  }

  template <typename D, size_t... Is>
  D MutateOp(D &&op, std::index_sequence<Is...>) const {
    return D(Mutate(std::move(op.template operand<Is>()))...);
  }

  template <typename T, size_t... Is>
  Extremum<T> MutateOp(Extremum<T> &&op, std::index_sequence<Is...>) const {
    return Extremum<T>(
        op.ordering, Mutate(std::move(op.template operand<Is>()))...);
  }

  template <int K, size_t... Is>
  ComplexComponent<K> MutateOp(
      ComplexComponent<K> &&op, std::index_sequence<Is...>) const {
    return ComplexComponent<K>(
        op.isImaginaryPart, Mutate(std::move(op.template operand<Is>()))...);
  }

  template <int K, size_t... Is>
  LogicalOperation<K> MutateOp(
      LogicalOperation<K> &&op, std::index_sequence<Is...>) const {
    return LogicalOperation<K>(
        op.logicalOperator, Mutate(std::move(op.template operand<Is>()))...);
  }

  Rewriter &rewriter_;
};

template <typename Rewriter> Mutator(Rewriter &) -> Mutator<Rewriter>;
} // namespace rewrite
} // namespace Fortran::evaluate

#endif // FORTRAN_EVALUATE_REWRITE_H_
