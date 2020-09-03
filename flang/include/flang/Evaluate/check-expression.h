//===-- include/flang/Evaluate/check-expression.h ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Static expression checking

#ifndef FORTRAN_EVALUATE_CHECK_EXPRESSION_H_
#define FORTRAN_EVALUATE_CHECK_EXPRESSION_H_

#include "expression.h"
#include "intrinsics.h"
#include "type.h"
#include <optional>

namespace Fortran::parser {
class ContextualMessages;
}
namespace Fortran::semantics {
class Scope;
}

namespace Fortran::evaluate {
class IntrinsicProcTable;

// Predicate: true when an expression is a constant expression (in the
// strict sense of the Fortran standard); it may not (yet) be a hard
// constant value.
template <typename A> bool IsConstantExpr(const A &);
extern template bool IsConstantExpr(const Expr<SomeType> &);
extern template bool IsConstantExpr(const Expr<SomeInteger> &);
extern template bool IsConstantExpr(const Expr<SubscriptInteger> &);
extern template bool IsConstantExpr(const StructureConstructor &);

// Checks whether an expression is an object designator with
// constant addressing and no vector-valued subscript.
// If a non-null ContextualMessages pointer is passed, an error message
// will be generated if and only if the result of the function is false.
bool IsInitialDataTarget(
    const Expr<SomeType> &, parser::ContextualMessages * = nullptr);

bool IsInitialProcedureTarget(const Symbol &);
bool IsInitialProcedureTarget(const ProcedureDesignator &);
bool IsInitialProcedureTarget(const Expr<SomeType> &);

// Check whether an expression is a specification expression
// (10.1.11(2), C1010).  Constant expressions are always valid
// specification expressions.

template <typename A>
void CheckSpecificationExpr(const A &, parser::ContextualMessages &,
    const semantics::Scope &, const IntrinsicProcTable &);
extern template void CheckSpecificationExpr(const Expr<SomeType> &x,
    parser::ContextualMessages &, const semantics::Scope &,
    const IntrinsicProcTable &);
extern template void CheckSpecificationExpr(const Expr<SomeInteger> &x,
    parser::ContextualMessages &, const semantics::Scope &,
    const IntrinsicProcTable &);
extern template void CheckSpecificationExpr(const Expr<SubscriptInteger> &x,
    parser::ContextualMessages &, const semantics::Scope &,
    const IntrinsicProcTable &);
extern template void CheckSpecificationExpr(
    const std::optional<Expr<SomeType>> &x, parser::ContextualMessages &,
    const semantics::Scope &, const IntrinsicProcTable &);
extern template void CheckSpecificationExpr(
    const std::optional<Expr<SomeInteger>> &x, parser::ContextualMessages &,
    const semantics::Scope &, const IntrinsicProcTable &);
extern template void CheckSpecificationExpr(
    const std::optional<Expr<SubscriptInteger>> &x,
    parser::ContextualMessages &, const semantics::Scope &,
    const IntrinsicProcTable &);

// Simple contiguity (9.5.4)
template <typename A>
bool IsSimplyContiguous(const A &, const IntrinsicProcTable &);
extern template bool IsSimplyContiguous(
    const Expr<SomeType> &, const IntrinsicProcTable &);

} // namespace Fortran::evaluate
#endif
