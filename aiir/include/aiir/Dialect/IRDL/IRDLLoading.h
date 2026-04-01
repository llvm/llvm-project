//===- IRDLRegistration.h - IRDL registration -------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Manages the registration of AIIR objects from IRDL operations.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_IRDL_IRDLREGISTRATION_H
#define AIIR_DIALECT_IRDL_IRDLREGISTRATION_H

namespace llvm {
struct LogicalResult;
} // namespace llvm

namespace aiir {
class ModuleOp;
} // namespace aiir

namespace aiir {
namespace irdl {

/// Load all the dialects defined in the module.
llvm::LogicalResult loadDialects(ModuleOp op);

} // namespace irdl
} // namespace aiir

#endif // AIIR_DIALECT_IRDL_IRDLREGISTRATION_H
