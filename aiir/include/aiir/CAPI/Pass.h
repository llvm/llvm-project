//===- IR.h - C API Utils for Core AIIR classes -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains declarations of implementation details of the C API for
// core AIIR classes. This file should not be included from C++ code other than
// C API implementation nor from C code.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CAPI_PASS_H
#define AIIR_CAPI_PASS_H

#include "aiir-c/Pass.h"

#include "aiir/CAPI/Wrap.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Pass/PassManager.h"

DEFINE_C_API_PTR_METHODS(AiirPass, aiir::Pass)
DEFINE_C_API_PTR_METHODS(AiirPassManager, aiir::PassManager)
DEFINE_C_API_PTR_METHODS(AiirOpPassManager, aiir::OpPassManager)

#endif // AIIR_CAPI_PASS_H
