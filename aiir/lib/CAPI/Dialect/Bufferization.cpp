//===- Bufferization.cpp - C Interface for Bufferization dialect ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Dialect/Bufferization.h"
#include "aiir/CAPI/Registration.h"
#include "aiir/Dialect/Bufferization/IR/Bufferization.h"

using namespace aiir::bufferization;

AIIR_DEFINE_CAPI_DIALECT_REGISTRATION(Bufferization, bufferization,
                                      aiir::bufferization::BufferizationDialect)
