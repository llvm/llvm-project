//===-- lib/Semantics/openmp-utils.h --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Common utilities used in OpenMP semantic checks.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_OPENMP_UTILS_H
#define FORTRAN_SEMANTICS_OPENMP_UTILS_H

#include "flang/Evaluate/type.h"
#include "flang/Parser/char-block.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/tools.h"

#include "llvm/ADT/ArrayRef.h"

#include <optional>
#include <string>

namespace Fortran::semantics {
class SemanticsContext;
class Symbol;

// Add this namespace to avoid potential conflicts
namespace omp {
std::string ThisVersion(unsigned version);
std::string TryVersion(unsigned version);

const parser::Designator *GetDesignatorFromObj(const parser::OmpObject &object);
const parser::DataRef *GetDataRefFromObj(const parser::OmpObject &object);
const parser::ArrayElement *GetArrayElementFromObj(
    const parser::OmpObject &object);
const Symbol *GetObjectSymbol(const parser::OmpObject &object);
const Symbol *GetArgumentSymbol(const parser::OmpArgument &argument);
std::optional<parser::CharBlock> GetObjectSource(
    const parser::OmpObject &object);

bool IsCommonBlock(const Symbol &sym);
bool IsExtendedListItem(const Symbol &sym);
bool IsVariableListItem(const Symbol &sym);
bool IsVarOrFunctionRef(const MaybeExpr &expr);

std::optional<SomeExpr> GetEvaluateExpr(const parser::Expr &parserExpr);
std::optional<evaluate::DynamicType> GetDynamicType(
    const parser::Expr &parserExpr);

std::optional<bool> IsContiguous(
    SemanticsContext &semaCtx, const parser::OmpObject &object);

std::vector<SomeExpr> GetAllDesignators(const SomeExpr &expr);
const SomeExpr *HasStorageOverlap(
    const SomeExpr &base, llvm::ArrayRef<SomeExpr> exprs);
bool IsSubexpressionOf(const SomeExpr &sub, const SomeExpr &super);
bool IsAssignment(const parser::ActionStmt *x);
bool IsPointerAssignment(const evaluate::Assignment &x);
const parser::Block &GetInnermostExecPart(const parser::Block &block);
} // namespace omp
} // namespace Fortran::semantics

#endif // FORTRAN_SEMANTICS_OPENMP_UTILS_H
