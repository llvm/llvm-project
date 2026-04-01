//===- MLProgram.cpp - C Interface for MLProgram dialect ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/MLProgram/IR/MLProgram.h"
#include "aiir-c/Dialect/MLProgram.h"
#include "aiir/CAPI/Registration.h"

AIIR_DEFINE_CAPI_DIALECT_REGISTRATION(MLProgram, ml_program,
                                      aiir::ml_program::MLProgramDialect)
