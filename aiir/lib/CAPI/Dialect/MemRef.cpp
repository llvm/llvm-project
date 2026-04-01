//===- MemRef.cpp - C Interface for MemRef dialect ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Dialect/MemRef.h"
#include "aiir/CAPI/Registration.h"
#include "aiir/Dialect/MemRef/IR/MemRef.h"

AIIR_DEFINE_CAPI_DIALECT_REGISTRATION(MemRef, memref,
                                      aiir::memref::MemRefDialect)
