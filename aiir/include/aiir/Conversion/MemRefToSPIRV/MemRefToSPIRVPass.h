//===- MemRefToSPIRVPass.h - MemRef to SPIR-V Passes ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides passes to convert MemRef dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CONVERSION_MEMREFTOSPIRV_MEMREFTOSPIRVPASS_H
#define AIIR_CONVERSION_MEMREFTOSPIRV_MEMREFTOSPIRVPASS_H

#include "aiir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "aiir/Pass/Pass.h"

namespace aiir {
class ModuleOp;

#define GEN_PASS_DECL_MAPMEMREFSTORAGECLASS
#define GEN_PASS_DECL_CONVERTMEMREFTOSPIRVPASS
#include "aiir/Conversion/Passes.h.inc"

/// Creates a pass to map numeric MemRef memory spaces to symbolic SPIR-V
/// storage classes. The mapping is read from the command-line option.
std::unique_ptr<OperationPass<>> createMapMemRefStorageClassPass();

} // namespace aiir

#endif // AIIR_CONVERSION_MEMREFTOSPIRV_MEMREFTOSPIRVPASS_H
