//===- CIROpInterfaces.td - CIR Op Interface Definitions --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CIR_OP_INTERFACES
#define MLIR_CIR_OP_INTERFACES

include "mlir/IR/OpBase.td"

let cppNamespace = "::cir" in {
  def CIRGlobalValueInterface
      : OpInterface<"CIRGlobalValueInterface"> {

    let methods = [
      InterfaceMethod<"",
      "bool", "hasAvailableExternallyLinkage", (ins), [{}],
      /*defaultImplementation=*/[{ return false; }]
      >,
      InterfaceMethod<"",
      "bool", "hasLocalLinkage", (ins), [{}],
      /*defaultImplementation=*/[{
        return cir::isLocalLinkage($_op.getLinkage());
      }]
      >,
      InterfaceMethod<"",
      "bool", "hasExternalWeakLinkage", (ins), [{}],
      /*defaultImplementation=*/[{
        return cir::isExternalWeakLinkage($_op.getLinkage());
      }]
      >,
      InterfaceMethod<"",
      "bool", "isExternalLinkage", (ins), [{}],
      /*defaultImplementation=*/[{
        return cir::isExternalLinkage($_op.getLinkage());
      }]
      >,
      InterfaceMethod<"",
      "bool", "isDeclarationForLinker", (ins), [{}],
      /*defaultImplementation=*/[{
        if ($_op.hasAvailableExternallyLinkage())
          return true;
        return $_op.isDeclaration();
      }]
      >,
      InterfaceMethod<"",
      "void", "setDSOLocal", (ins "bool":$val), [{}],
      /*defaultImplementation=*/[{
        $_op.setDsolocal(val);
      }]
      >,
    ];
    let extraClassDeclaration = [{
    bool canBenefitFromLocalAlias();
  }];
  }

} // namespace cir

#endif // MLIR_CIR_OP_INTERFACES
