//===-- aiir-c/RegisterEverything.h - Register all AIIR entities --*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This header contains registration entry points for AIIR upstream dialects
// and passes. Downstream projects typically will not want to use this unless
// if they don't care about binary size or build bloat and just wish access
// to the entire set of upstream facilities. For those that do care, they
// should use registration functions specific to their project.
//===----------------------------------------------------------------------===//

#ifndef AIIR_C_REGISTER_EVERYTHING_H
#define AIIR_C_REGISTER_EVERYTHING_H

#include "aiir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Appends all upstream dialects and extensions to the dialect registry.
AIIR_CAPI_EXPORTED void aiirRegisterAllDialects(AiirDialectRegistry registry);

/// Register all translations to LLVM IR for dialects that can support it.
AIIR_CAPI_EXPORTED void aiirRegisterAllLLVMTranslations(AiirContext context);

/// Register all compiler passes of AIIR.
AIIR_CAPI_EXPORTED void aiirRegisterAllPasses(void);

#ifdef __cplusplus
}
#endif

#endif // AIIR_C_REGISTER_EVERYTHING_H
