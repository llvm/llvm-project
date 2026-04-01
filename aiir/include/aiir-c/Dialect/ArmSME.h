//===-- aiir-c/Dialect/ArmSME.h - C API for ArmSME Dialect --------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_C_DIALECT_ARMSME_H
#define AIIR_C_DIALECT_ARMSME_H

#include "aiir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

AIIR_DECLARE_CAPI_DIALECT_REGISTRATION(ArmSME, arm_sme);

#ifdef __cplusplus
}
#endif

#include "aiir/Dialect/ArmSME/Transforms/Passes.capi.h.inc"

#endif // AIIR_C_DIALECT_ARMSME_H
