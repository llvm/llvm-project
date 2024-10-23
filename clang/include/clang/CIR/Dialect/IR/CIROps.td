//===-- CIROps.td - CIR dialect definition -----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of the CIR dialect
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_DIALECT_IR_CIROPS
#define LLVM_CLANG_CIR_DIALECT_IR_CIROPS

include "clang/CIR/Dialect/IR/CIRDialect.td"

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"

//===----------------------------------------------------------------------===//
// CIR Ops
//===----------------------------------------------------------------------===//

class LLVMLoweringInfo {
  string llvmOp = "";
}

class CIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<CIR_Dialect, mnemonic, traits>, LLVMLoweringInfo;

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

// For starters, cir.func has only name, nothing else.  The other properties
// of a function will be added over time as more of ClangIR is upstreamed.

def FuncOp : CIR_Op<"func"> {
  let summary = "Declare or define a function";
  let description = [{
    ... lots of text to be added later ...
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);

  let skipDefaultBuilders = 1;

  let builders = [OpBuilder<(ins "StringRef":$name)>];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

#endif // LLVM_CLANG_CIR_DIALECT_IR_CIROPS
