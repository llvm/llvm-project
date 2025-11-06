//===- FIROpenACCUtils.h - FIR OpenACC Utilities ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares utility functions for FIR OpenACC support.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_OPENACC_SUPPORT_FIROPENACCUTILS_H
#define FORTRAN_OPTIMIZER_OPENACC_SUPPORT_FIROPENACCUTILS_H

#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/Value.h"
#include <string>

namespace fir {
namespace acc {

/// Attempts to extract the variable name from a value by walking through
/// FIR operations and looking for variable names.
/// \param v The value to extract the variable name from
/// \param preferDemangledName If true, prefers demangled/bindc names over
///        mangled/unique names. If false, prefers mangled names.
/// Returns empty string if no name is found.
std::string getVariableName(mlir::Value v, bool preferDemangledName = true);

/// Get the recipe name for a given recipe kind, FIR type, and optional
/// variable. Uses FIR's type string representation with appropriate prefix. For
/// firstprivate and reduction recipes, handles bounds suffix when all bounds
/// are constant. For reduction recipes, embeds the operator name in the recipe.
/// \param kind The recipe kind (private, firstprivate, or reduction)
/// \param type The FIR type (must be a FIR type)
/// \param var Optional variable value
/// \param bounds Optional bounds for array sections (used for suffix
/// generation)
/// \param reductionOp Optional reduction operator (required for reduction
/// recipes)
/// \return The complete recipe name with all necessary suffixes
std::string getRecipeName(mlir::acc::RecipeKind kind, mlir::Type type,
                          mlir::Value var = nullptr,
                          llvm::ArrayRef<mlir::Value> bounds = {},
                          mlir::acc::ReductionOperator reductionOp =
                              mlir::acc::ReductionOperator::AccNone);

/// Check if all bounds are expressed with constant values.
/// \param bounds Array of DataBoundsOp values to check
/// \return true if all bounds have constant lowerbound/upperbound or extent
bool areAllBoundsConstant(llvm::ArrayRef<mlir::Value> bounds);

} // namespace acc
} // namespace fir

#endif // FORTRAN_OPTIMIZER_OPENACC_SUPPORT_FIROPENACCUTILS_H
