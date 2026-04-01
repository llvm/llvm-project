//===- ArmSVEPasses.cpp - C API for ArmSVE Dialect Passes -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/CAPI/Pass.h"
#include "aiir/Dialect/ArmSVE/Transforms/Passes.h"
#include "aiir/Pass/Pass.h"

// Must include the declarations as they carry important visibility attributes.
#include "aiir/Dialect/ArmSVE/Transforms/Passes.capi.h.inc"
using namespace aiir;
using namespace aiir::arm_sve;

#ifdef __cplusplus
extern "C" {
#endif

#include "aiir/Dialect/ArmSVE/Transforms/Passes.capi.cpp.inc"

#ifdef __cplusplus
}
#endif
