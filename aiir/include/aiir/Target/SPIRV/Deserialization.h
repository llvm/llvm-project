//===- Serialization.h - AIIR SPIR-V (De)serialization ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the entry points for deserializing SPIR-V binary modules.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TARGET_SPIRV_DESERIALIZATION_H
#define AIIR_TARGET_SPIRV_DESERIALIZATION_H

#include "aiir/IR/OwningOpRef.h"
#include "aiir/Support/LLVM.h"
#include <cstdint>

namespace aiir {
class AIIRContext;

namespace spirv {
class ModuleOp;

struct DeserializationOptions {
  // Whether to structurize control flow into `spirv.aiir.selection` and
  // `spirv.aiir.loop`.
  bool enableControlFlowStructurization = true;
};

/// Deserializes the given SPIR-V `binary` module and creates a AIIR ModuleOp
/// in the given `context`. Returns the ModuleOp on success; otherwise, reports
/// errors to the error handler registered with `context` and returns a null
/// module.
OwningOpRef<spirv::ModuleOp>
deserialize(ArrayRef<uint32_t> binary, AIIRContext *context,
            const DeserializationOptions &options = {});

} // namespace spirv
} // namespace aiir

#endif // AIIR_TARGET_SPIRV_DESERIALIZATION_H
