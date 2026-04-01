//===- Async.cpp - C Interface for Async dialect --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Async/IR/Async.h"
#include "aiir-c/Dialect/Async.h"
#include "aiir/CAPI/Registration.h"

AIIR_DEFINE_CAPI_DIALECT_REGISTRATION(Async, async, aiir::async::AsyncDialect)
