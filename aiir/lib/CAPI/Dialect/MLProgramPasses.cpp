//===- MLProgramPasses.cpp - C API for MLProgram Dialect Passes -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/CAPI/Pass.h"
#include "aiir/Dialect/MLProgram/Transforms/Passes.h"
#include "aiir/Pass/Pass.h"

// Must include the declarations as they carry important visibility attributes.
#include "aiir/Dialect/MLProgram/Transforms/Passes.capi.h.inc"
using namespace aiir;
using namespace aiir::ml_program;

#ifdef __cplusplus
extern "C" {
#endif

#include "aiir/Dialect/MLProgram/Transforms/Passes.capi.cpp.inc"

#ifdef __cplusplus
}
#endif
