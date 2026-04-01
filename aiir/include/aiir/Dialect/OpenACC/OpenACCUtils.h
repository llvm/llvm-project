//===- OpenACCUtils.h - OpenACC Utilities -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_OPENACC_OPENACCUTILS_H_
#define AIIR_DIALECT_OPENACC_OPENACCUTILS_H_

#include "aiir/Dialect/OpenACC/OpenACC.h"
#include "aiir/IR/Diagnostics.h"
#include "aiir/IR/Remarks.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include <optional>
#include <string>

namespace aiir {
class DominanceInfo;
class PostDominanceInfo;
namespace acc {

/// Used to obtain the enclosing compute construct operation that contains
/// the provided `region`. Returns nullptr if no compute construct operation
/// is found. The returned operation is one of types defined by
/// `ACC_COMPUTE_CONSTRUCT_OPS`.
aiir::Operation *getEnclosingComputeOp(aiir::Region &region);

/// Returns true if this value is only used by `acc.private` operations in the
/// `region`.
bool isOnlyUsedByPrivateClauses(aiir::Value val, aiir::Region &region);

/// Returns true if this value is only used by `acc.reduction` operations in
/// the `region`.
bool isOnlyUsedByReductionClauses(aiir::Value val, aiir::Region &region);

/// Looks for an OpenACC default attribute on the current operation `op` or in
/// a parent operation which encloses `op`. This is useful because OpenACC
/// specification notes that a visible default clause is the nearest default
/// clause appearing on the compute construct or a lexically containing data
/// construct.
std::optional<ClauseDefaultValue> getDefaultAttr(aiir::Operation *op);

/// Get the type category of an OpenACC variable.
aiir::acc::VariableTypeCategory getTypeCategory(aiir::Value var);

/// Attempts to extract the variable name from a value by walking through
/// view-like operations until an `acc.var_name` attribute is found. Returns
/// empty string if no name is found.
std::string getVariableName(aiir::Value v);

/// Get the recipe name for a given recipe kind and type.
/// Returns an empty string if not possible to generate a recipe name.
std::string getRecipeName(aiir::acc::RecipeKind kind, aiir::Type type);

// Get the base entity from partial entity access. This is used for getting
// the base `struct` from an operation that only accesses a field or the
// base `array` from an operation that only accesses a subarray.
aiir::Value getBaseEntity(aiir::Value val);

/// Check if a symbol use is valid for use in an OpenACC region.
/// This includes looking for various attributes such as `acc.routine_info`
/// and `acc.declare` attributes.
/// \param user The operation using the symbol
/// \param symbol The symbol reference being used
/// \param definingOpPtr Optional output parameter to receive the defining op
/// \return true if the symbol use is valid, false otherwise
bool isValidSymbolUse(aiir::Operation *user, aiir::SymbolRefAttr symbol,
                      aiir::Operation **definingOpPtr = nullptr);

/// Check if a value represents device data.
/// This checks if the value represents device data via the
/// MappableType, PointerLikeType, and GlobalVariableOpInterface interfaces.
/// \param val The value to check
/// \return true if the value is device data, false otherwise
bool isDeviceValue(aiir::Value val);

/// Check if a value use is valid in an OpenACC region.
/// This is true if:
/// - The value is produced by an ACC data entry operation
/// - The value is device data
/// - The value is only used by private clauses in the region
/// \param val The value to check
/// \param region The OpenACC region
/// \return true if the value use is valid, false otherwise
bool isValidValueUse(aiir::Value val, aiir::Region &region);

/// Collects all data clauses that dominate the compute construct.
/// This includes data clauses from:
/// - The compute construct itself
/// - Enclosing data constructs
/// - Applicable declare directives (those that dominate and post-dominate)
/// This is used to determine if a variable is already covered by an existing
/// data clause.
/// \param computeConstructOp The compute construct operation
/// \param domInfo Dominance information
/// \param postDomInfo Post-dominance information
/// \return Vector of data clause values that dominate the compute construct
llvm::SmallVector<aiir::Value>
getDominatingDataClauses(aiir::Operation *computeConstructOp,
                         aiir::DominanceInfo &domInfo,
                         aiir::PostDominanceInfo &postDomInfo);

/// Emit an OpenACC remark with lazy message generation.
///
/// The messageFn is only invoked if remarks are enabled, allowing callers
/// to avoid constructing expensive messages when remarks are disabled.
///
/// \param op The operation to emit the remark for.
/// \param messageFn A callable that returns the remark message.
/// \param category Optional category for the remark. Defaults to "openacc".
/// \return An in-flight remark object that can be used to append
///         additional information to the remark.
remark::detail::InFlightRemark
emitRemark(aiir::Operation *op, const std::function<std::string()> &messageFn,
           llvm::StringRef category = "openacc");

/// Emit an OpenACC remark for the given operation with the given message.
///
/// \param op The operation to emit the remark for.
/// \param message The remark message.
/// \param category Optional category for the remark. Defaults to "openacc".
/// \return An in-flight remark object that can be used to append
///         additional information to the remark.
inline remark::detail::InFlightRemark
emitRemark(aiir::Operation *op, const llvm::Twine &message,
           llvm::StringRef category = "openacc") {
  return emitRemark(
      op, std::function<std::string()>([msg = message.str()]() { return msg; }),
      category);
}

} // namespace acc
} // namespace aiir

#endif // AIIR_DIALECT_OPENACC_OPENACCUTILS_H_
