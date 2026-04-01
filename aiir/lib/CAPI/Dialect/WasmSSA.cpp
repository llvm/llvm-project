//===- WasmSSA.cpp - C Interface for WasmSSA dialect ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Dialect/WasmSSA.h"
#include "aiir/CAPI/Registration.h"
#include "aiir/Dialect/WasmSSA/IR/WasmSSA.h"

AIIR_DEFINE_CAPI_DIALECT_REGISTRATION(WasmSSA, wasmssa,
                                      aiir::wasmssa::WasmSSADialect)
