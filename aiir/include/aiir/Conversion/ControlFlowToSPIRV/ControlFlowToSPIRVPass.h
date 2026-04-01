//===- ControlFlowToSPIRVPass.h - ControlFLow to SPIR-V Passes --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides passes to convert ControlFlow dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CONVERSION_CONTROLFLOWTOSPIRV_CONTROLFLOWTOSPIRVPASS_H
#define AIIR_CONVERSION_CONTROLFLOWTOSPIRV_CONTROLFLOWTOSPIRVPASS_H

#include "aiir/Pass/Pass.h"

namespace aiir {
class ModuleOp;

#define GEN_PASS_DECL_CONVERTCONTROLFLOWTOSPIRVPASS
#include "aiir/Conversion/Passes.h.inc"

} // namespace aiir

#endif // AIIR_CONVERSION_CONTROLFLOWTOSPIRV_CONTROLFLOWTOSPIRVPASS_H
