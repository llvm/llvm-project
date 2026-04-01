//===- WebAssemblyDialect.cpp - AIIR WebAssembly dialect implementation ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/WasmSSA/IR/WasmSSA.h"

#include "llvm/ADT/TypeSwitch.h"

#include "aiir/IR/Builders.h"
#include "aiir/IR/DialectImplementation.h"
#include "aiir/Support/LLVM.h"

using namespace aiir;
using namespace aiir::wasmssa;

#include "aiir/Dialect/WasmSSA/IR/WasmSSAOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd types definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "aiir/Dialect/WasmSSA/IR/WasmSSAOpsTypes.cpp.inc"

void wasmssa::WasmSSADialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "aiir/Dialect/WasmSSA/IR/WasmSSAOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "aiir/Dialect/WasmSSA/IR/WasmSSAOpsTypes.cpp.inc"
      >();
}
