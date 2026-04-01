//===- VerificationUtils.h - Common verification utilities ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines common verification utilities that can be shared
// across multiple AIIR dialects. These utilities help reduce code duplication
// for common verification patterns.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_UTILS_VERIFICATIONUTILS_H
#define AIIR_DIALECT_UTILS_VERIFICATIONUTILS_H

#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/Operation.h"
#include "aiir/Support/LLVM.h"

namespace aiir {

/// Verify that the number of dynamic size operands matches the number of
/// dynamic dimensions in the shaped type. Returns failure and emits an error
/// if the counts don't match.
LogicalResult verifyDynamicDimensionCount(Operation *op, ShapedType type,
                                          ValueRange dynamicSizes);

/// Verify that two shaped types have matching ranks. Returns failure and emits
/// an error if ranks don't match. Unranked types are considered compatible.
LogicalResult verifyRanksMatch(Operation *op, ShapedType lhs, ShapedType rhs,
                               StringRef lhsName, StringRef rhsName);

/// Verify that two shaped types have matching element types. Returns failure
/// and emits an error if element types don't match.
LogicalResult verifyElementTypesMatch(Operation *op, ShapedType lhs,
                                      ShapedType rhs, StringRef lhsName,
                                      StringRef rhsName);

} // namespace aiir

#endif // AIIR_DIALECT_UTILS_VERIFICATIONUTILS_H
