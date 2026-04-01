//=- VectorToSPIRVPass.h - Vector to SPIR-V Passes ----------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides passes to convert Vector dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CONVERSION_VECTORTOSPIRV_VECTORTOSPIRVPASS_H
#define AIIR_CONVERSION_VECTORTOSPIRV_VECTORTOSPIRVPASS_H

#include "aiir/Pass/Pass.h"

namespace aiir {
class ModuleOp;

#define GEN_PASS_DECL_CONVERTVECTORTOSPIRVPASS
#include "aiir/Conversion/Passes.h.inc"

} // namespace aiir

#endif // AIIR_CONVERSION_VECTORTOSPIRV_VECTORTOSPIRVPASS_H
