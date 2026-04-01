//===-- WasmSSAInterfaces.td - WasmSSA Interfaces -*- tablegen -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines interfaces for the WasmSSA dialect in AIIR.
//
//===----------------------------------------------------------------------===//

#ifndef WasmSSA_INTERFACES
#define WasmSSA_INTERFACES

include "aiir/IR/OpBase.td"
include "aiir/IR/BuiltinAttributes.td"

def LabelLevelOpInterface : OpInterface<"LabelLevelOpInterface"> {
  let cppNamespace = "::aiir::wasmssa";
  let description = [{
    Operation that defines one level of nesting for Wasm branching.

    These ops defines Wasm control flow nesting levels (Wasm Labels) that Wasm
    branching operations can target.
    The branching operations specify a number of nesting level they want to exit,
    and are redirected to the target of the corresponding nesting LabelLevelOp.

    As multiple level can be escaped at once, the level defining ops need themselves
    to be `Terminator` ops.
  }];
  let methods = [
    InterfaceMethod<
      /*desc=*/        "Returns the target block address",
      /*returnType=*/  "::aiir::Block*",
      /*methodName=*/  "getLabelTarget",
      /*args=*/        (ins)
    >
  ];

  let verify = [{
    return success(
      succeeded(verifyLabelLevelInterfaceIsTerminator<ConcreteOp>()) &&
      succeeded(verifyLabelLevelInterface($_op)));
  }];
}

def LabelBranchingOpInterface : OpInterface<"LabelBranchingOpInterface"> {
  let cppNamespace = "::aiir::wasmssa";
  let description = [{
    Wasm operation that targets a label for a jump.
  }];
  let methods = [
    InterfaceMethod<
      /*desc=*/        "Returns the number of context to break from",
      /*returnType=*/  "size_t",
      /*methodName=*/  "getExitLevel",
      /*args=*/        (ins)
    >,
    InterfaceMethod<
      /*desc=*/        "Returns the destination of this operation",
      /*returnType=*/  "LabelLevelOpInterface",
      /*methodName=*/  "getTargetOp",
      /*args=*/        (ins),
      /*methodBody=*/ [{
        return *LabelBranchingOpInterface::getTargetOpFromBlock($_op.getOperation()->getBlock(), $_op.getExitLevel());
      }]
    >,
    InterfaceMethod<
      /*desc=*/        "Return the target control flow ops that defined the label of this operation",
      /*returnType=*/  "::aiir::Block*",
      /*methodName=*/  "getTarget",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        auto op = aiir::cast<LabelBranchingOpInterface>(this->getOperation());
        return op.getTargetOp().getLabelTarget();
      }]
    >
  ];

  let extraClassDeclaration = [{
    static ::llvm::FailureOr<LabelLevelOpInterface> getTargetOpFromBlock(::aiir::Block *block, uint32_t level);
  }];
  let verify = [{return verifyLabelBranchingOpInterface($_op);}];
}

def ImportOpInterface : OpInterface<"ImportOpInterface"> {
  let cppNamespace = "::aiir::wasmssa";
  let description = [{
    Operation that imports a symbol from an external Wasm module;
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/        "Returns the module name for the import",
      /*returnType=*/  "::llvm::StringRef",
      /*methodName=*/  "getModuleName",
      /*args=*/        (ins)
      >,
    InterfaceMethod<
      /*desc=*/        "Returns the import name for the import",
      /*returnType=*/  "::llvm::StringRef",
      /*methodName=*/  "getImportName",
      /*args=*/        (ins)
      >,
    InterfaceMethod<
      /*desc=*/        "Returns the Wasm index based symbol of the op",
      /*returnType=*/  "::aiir::StringAttr",
      /*methodName=*/  "getSymbolName",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        auto op = aiir::cast<ConcreteOp>(this->getOperation());
        return op.getSymNameAttr();
      }]
      >,
    InterfaceMethod<
      /*desc=*/        "Returns the qualified name of the import",
      /*returnType=*/  "std::string",
      /*methodName=*/  "getQualifiedImportName",
      /*args=*/        (ins),
      /*methodBody=*/  [{
        return ($_op.getModuleName() + ::llvm::Twine{"::"} + $_op.getImportName()).str();
      }]
      >,
  ];
}

def ConstantExpressionInitializerOpTrait : NativeOpTrait<"ConstantExpressionInitializerOpTrait"> {
  let cppNamespace = "::aiir::wasmssa";
}

def ConstantExprOpTrait : NativeOpTrait<"ConstantExprOpTrait"> {
  let cppNamespace = "::aiir::wasmssa";
}

#endif // WEBASSEMBLY_INTERFACES
