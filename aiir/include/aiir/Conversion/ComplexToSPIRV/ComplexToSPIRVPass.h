//===- ComplexToSPIRVPass.h - Complex to SPIR-V Passes ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides passes to convert Complex dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CONVERSION_COMPLEXTOSPIRV_COMPLEXTOSPIRVPASS_H
#define AIIR_CONVERSION_COMPLEXTOSPIRV_COMPLEXTOSPIRVPASS_H

#include "aiir/Pass/Pass.h"

namespace aiir {
class ModuleOp;

#define GEN_PASS_DECL_CONVERTCOMPLEXTOSPIRVPASS
#include "aiir/Conversion/Passes.h.inc"

} // namespace aiir

#endif // AIIR_CONVERSION_COMPLEXTOSPIRV_COMPLEXTOSPIRVPASS_H
