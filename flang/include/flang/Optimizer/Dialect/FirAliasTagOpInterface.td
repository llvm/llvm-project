//===-- FirAliasTagOpInterface.td --------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "mlir/IR/Interfaces.td"

def FirAliasTagOpInterface : OpInterface<"FirAliasTagOpInterface"> {
  let description = [{
    An interface for memory operations that can carry alias analysis metadata.
    It provides setters and getters for the operation's alias analysis
    attributes. The default implementations of the interface methods expect
    the operation to have an attribute of type ArrayAttr named tbaa.
    Unlike the mlir::LLVM::AliasAnalysisOpInterface, this only supports tbaa.
  }];

  let cppNamespace = "::fir";
  let verify = [{ return detail::verifyFirAliasTagOpInterface($_op); }];

  let methods = [
    InterfaceMethod<
      /*desc=*/        "Returns the tbaa attribute or nullptr",
      /*returnType=*/  "mlir::ArrayAttr",
      /*methodName=*/  "getTBAATagsOrNull",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        auto op = mlir::cast<ConcreteOp>(this->getOperation());
        return op.getTbaaAttr();
      }]
      >,
    InterfaceMethod<
      /*desc=*/        "Sets the tbaa attribute",
      /*returnType=*/  "void",
      /*methodName=*/  "setTBAATags",
      /*args=*/        (ins "const mlir::ArrayAttr":$attr),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        auto op = mlir::cast<ConcreteOp>(this->getOperation());
        op.setTbaaAttr(attr);
      }]
      >,
    InterfaceMethod<
      /*desc=*/        "Returns a list of all pointer operands accessed by the "
                       "operation",
      /*returnType=*/  "::llvm::SmallVector<::mlir::Value>",
      /*methodName=*/  "getAccessedOperands",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        auto op = mlir::cast<ConcreteOp>(this->getOperation());
        return {op.getMemref()};
      }]
      >
  ];
}
