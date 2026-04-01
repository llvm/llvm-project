//===- ControlFlow.cpp - C Interface for ControlFlow dialect --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Dialect/ControlFlow.h"
#include "aiir/CAPI/Registration.h"
#include "aiir/Dialect/ControlFlow/IR/ControlFlow.h"

AIIR_DEFINE_CAPI_DIALECT_REGISTRATION(ControlFlow, cf,
                                      aiir::cf::ControlFlowDialect)
