//===- WasmSSA.h - WasmSSA dialect ------------------*- C++-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_WasmSSA_IR_WasmSSA_H_
#define AIIR_DIALECT_WasmSSA_IR_WasmSSA_H_

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/IR/Dialect.h"

//===----------------------------------------------------------------------===//
// WebAssemblyDialect
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/WasmSSA/IR/WasmSSAOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// WebAssembly Dialect Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "aiir/Dialect/WasmSSA/IR/WasmSSAOpsTypes.h.inc"

//===----------------------------------------------------------------------===//
// WebAssembly Interfaces
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/WasmSSA/IR/WasmSSAInterfaces.h"

//===----------------------------------------------------------------------===//
// WebAssembly Dialect Operations
//===----------------------------------------------------------------------===//
#include "aiir/IR/SymbolTable.h"
#include "aiir/Interfaces/CallInterfaces.h"
#include "aiir/Interfaces/FunctionInterfaces.h"
#include "aiir/Interfaces/InferTypeOpInterface.h"

//===----------------------------------------------------------------------===//
// WebAssembly Constraints
//===----------------------------------------------------------------------===//

namespace aiir {
namespace wasmssa {
#include "aiir/Dialect/WasmSSA/IR/WasmSSATypeConstraints.h.inc"
}
} // namespace aiir

#define GET_OP_CLASSES
#include "aiir/Dialect/WasmSSA/IR/WasmSSAOps.h.inc"

#endif // AIIR_DIALECT_WasmSSA_IR_WasmSSA_H_
