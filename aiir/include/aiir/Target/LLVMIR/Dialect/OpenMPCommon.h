//===- OpenMPCommon.h - Utils for translating AIIR dialect to LLVM IR------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines general utilities for AIIR Dialect translations to LLVM IR.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TARGET_LLVMIR_DIALECT_OPENMPCOMMON_H
#define AIIR_TARGET_LLVMIR_DIALECT_OPENMPCOMMON_H

#include "aiir/IR/BuiltinAttributes.h"
#include "aiir/IR/Location.h"
#include "aiir/Support/LLVM.h"

#include "llvm/Frontend/OpenMP/OMPIRBuilder.h"
#include "llvm/IR/IRBuilder.h"

namespace aiir {
namespace LLVM {

/// Create a constant string location from the AIIR Location information.
llvm::Constant *createSourceLocStrFromLocation(Location loc,
                                               llvm::OpenMPIRBuilder &builder,
                                               StringRef name,
                                               uint32_t &strLen);

/// Create a constant string representing the mapping information extracted from
/// the AIIR location information.
llvm::Constant *createMappingInformation(Location loc,
                                         llvm::OpenMPIRBuilder &builder);

} // namespace LLVM
} // namespace aiir

#endif // AIIR_TARGET_LLVMIR_DIALECT_OPENMPCOMMON_H
