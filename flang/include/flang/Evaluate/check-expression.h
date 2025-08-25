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

// Predicate: true when an expression is a constant expression (in the
// strict sense of the Fortran standard); it may not (yet) be a hard
// constant value.
template <typename A> bool IsConstantExpr(const A &);
extern template bool IsConstantExpr(const Expr<SomeType> &);
extern template bool IsConstantExpr(const Expr<SomeInteger> &);
extern template bool IsConstantExpr(const Expr<SubscriptInteger> &);
extern template bool IsConstantExpr(const StructureConstructor &);

// Predicate: true when an expression is a constant expression (in the
// strict sense of the Fortran standard) or a dummy argument with
// INTENT(IN) and no VALUE.  This is useful for representing explicit
// shapes of other dummy arguments.
template <typename A> bool IsScopeInvariantExpr(const A &);
extern template bool IsScopeInvariantExpr(const Expr<SomeType> &);
extern template bool IsScopeInvariantExpr(const Expr<SomeInteger> &);
extern template bool IsScopeInvariantExpr(const Expr<SubscriptInteger> &);

// Predicate: true when an expression actually is a typed Constant<T>,
// perhaps with parentheses and wrapping around it.  False for all typeless
// expressions, including BOZ literals.
template <typename A> bool IsActuallyConstant(const A &);
extern template bool IsActuallyConstant(const Expr<SomeType> &);
extern template bool IsActuallyConstant(const Expr<SomeInteger> &);
extern template bool IsActuallyConstant(const Expr<SubscriptInteger> &);
extern template bool IsActuallyConstant(
    const std::optional<Expr<SubscriptInteger>> &);

// Checks whether an expression is an object designator with
// constant addressing and no vector-valued subscript.
// If a non-null ContextualMessages pointer is passed, an error message
// will be generated if and only if the result of the function is false.
bool IsInitialDataTarget(
    const Expr<SomeType> &, parser::ContextualMessages * = nullptr);

bool IsInitialProcedureTarget(const Symbol &);
bool IsInitialProcedureTarget(const ProcedureDesignator &);
bool IsInitialProcedureTarget(const Expr<SomeType> &);

// Emit warnings about default REAL literal constants in contexts that
// will be converted to a higher precision REAL kind than the default.
void CheckRealWidening(
    const Expr<SomeType> &, const DynamicType &toType, FoldingContext &);
void CheckRealWidening(const Expr<SomeType> &,
    const std::optional<DynamicType> &, FoldingContext &);

// Validate the value of a named constant, the static initial
// value of a non-pointer non-allocatable non-dummy variable, or the
// default initializer of a component of a derived type (or instantiation
// of a derived type).  Converts type and expands scalars as necessary.
std::optional<Expr<SomeType>> NonPointerInitializationExpr(const Symbol &,
    Expr<SomeType> &&, FoldingContext &,
    const semantics::Scope *instantiation = nullptr);

// Check whether an expression is a specification expression
// (10.1.11(2), C1010).  Constant expressions are always valid
// specification expressions.

template <typename A>
void CheckSpecificationExpr(const A &, const semantics::Scope &,
    FoldingContext &, bool forElementalFunctionResult);
extern template void CheckSpecificationExpr(const Expr<SomeType> &x,
    const semantics::Scope &, FoldingContext &,
    bool forElementalFunctionResult);
extern template void CheckSpecificationExpr(const Expr<SomeInteger> &x,
    const semantics::Scope &, FoldingContext &,
    bool forElementalFunctionResult);
extern template void CheckSpecificationExpr(const Expr<SubscriptInteger> &x,
    const semantics::Scope &, FoldingContext &,
    bool forElementalFunctionResult);
extern template void CheckSpecificationExpr(
    const std::optional<Expr<SomeType>> &x, const semantics::Scope &,
    FoldingContext &, bool forElementalFunctionResult);
extern template void CheckSpecificationExpr(
    const std::optional<Expr<SomeInteger>> &x, const semantics::Scope &,
    FoldingContext &, bool forElementalFunctionResult);
extern template void CheckSpecificationExpr(
    const std::optional<Expr<SubscriptInteger>> &x, const semantics::Scope &,
    FoldingContext &, bool forElementalFunctionResult);

// Contiguity & "simple contiguity" (9.5.4)
// Named constant sections are expressions, and as such their evaluation is
// considered to be contiguous. This avoids funny situations where
// IS_CONTIGUOUS(cst(1:10:2)) would fold to true because `cst(1:10:2)` is
// folded into an array constructor literal, but IS_CONTIGUOUS(cst(i:i+9:2))
// folds to false because the named constant reference cannot be folded.
// Note that these IS_CONTIGUOUS usages are not portable (can probably be
// considered to fall into F2023 8.5.7 (4)), and existing compilers are not
// consistent here.
// However, the compiler may very well decide to create a descriptor over
// `cst(i:i+9:2)` when it can to avoid copies, and as such it needs internally
// to be able to tell the actual contiguity of that array section over the
// read-only data.
template <typename A>
std::optional<bool> IsContiguous(const A &, FoldingContext &,
    bool namedConstantSectionsAreContiguous = true,
    bool firstDimensionStride1 = false);
extern template std::optional<bool> IsContiguous(const Expr<SomeType> &,
    FoldingContext &, bool namedConstantSectionsAreContiguous,
    bool firstDimensionStride1);
extern template std::optional<bool> IsContiguous(const ActualArgument &,
    FoldingContext &, bool namedConstantSectionsAreContiguous,
    bool firstDimensionStride1);
extern template std::optional<bool> IsContiguous(const ArrayRef &,
    FoldingContext &, bool namedConstantSectionsAreContiguous,
    bool firstDimensionStride1);
extern template std::optional<bool> IsContiguous(const Substring &,
    FoldingContext &, bool namedConstantSectionsAreContiguous,
    bool firstDimensionStride1);
extern template std::optional<bool> IsContiguous(const Component &,
    FoldingContext &, bool namedConstantSectionsAreContiguous,
    bool firstDimensionStride1);
extern template std::optional<bool> IsContiguous(const ComplexPart &,
    FoldingContext &, bool namedConstantSectionsAreContiguous,
    bool firstDimensionStride1);
extern template std::optional<bool> IsContiguous(const CoarrayRef &,
    FoldingContext &, bool namedConstantSectionsAreContiguous,
    bool firstDimensionStride1);
extern template std::optional<bool> IsContiguous(const Symbol &,
    FoldingContext &, bool namedConstantSectionsAreContiguous,
    bool firstDimensionStride1);
static inline std::optional<bool> IsContiguous(const SymbolRef &s,
    FoldingContext &c, bool namedConstantSectionsAreContiguous = true) {
  return IsContiguous(s.get(), c, namedConstantSectionsAreContiguous);
}
template <typename A>
bool IsSimplyContiguous(const A &x, FoldingContext &context,
    bool namedConstantSectionsAreContiguous = true) {
  return IsContiguous(x, context, namedConstantSectionsAreContiguous)
      .value_or(false);
}

template <typename A> bool IsErrorExpr(const A &);
extern template bool IsErrorExpr(const Expr<SomeType> &);

std::optional<parser::Message> CheckStatementFunction(
    const Symbol &, const Expr<SomeType> &, FoldingContext &);

bool MayNeedCopy(const ActualArgument *, const characteristics::DummyArgument *,
    FoldingContext &, bool forCopyOut);

} // namespace Fortran::evaluate
#endif
