//===- CIROpInterfaces.td - CIR Op Interface Definitions --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface to CIR operations.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_INTERFACES_CIROPINTERFACES_TD
#define CLANG_CIR_INTERFACES_CIROPINTERFACES_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"

let cppNamespace = "::cir" in {
  // The CIRCallOpInterface must be used instead of CallOpInterface when looking
  // at arguments and other bits of CallOp. This creates a level of abstraction
  // that's useful for handling indirect calls and other details.
  def CIRCallOpInterface : OpInterface<"CIRCallOpInterface", [CallOpInterface]> {
    // Currently we don't have any methods defined in CIRCallOpInterface. We'll
    // add more methods as the upstreaming proceeds.
    let methods = [
      InterfaceMethod<
          "Return the operand at index 'i', accounts for indirect call or "
          "exception info",
          "mlir::Value", "getArgOperand",
          (ins "unsigned":$i)>,
      InterfaceMethod<
          "Return the number of operands, accounts for indirect call or "
          "exception info",
          "unsigned", "getNumArgOperands", (ins)>,
      InterfaceMethod<"Return whether the callee is nothrow",
                      "bool", "getNothrow", (ins)>,
      InterfaceMethod<"Return the side effects of the call operation",
                      "cir::SideEffect", "getSideEffect", (ins)>,
    ];
  }

  def CIRGlobalValueInterface
      : OpInterface<"CIRGlobalValueInterface", [Symbol]> {

    let methods = [
      InterfaceMethod<"",
      "bool", "hasExternalLinkage", (ins), [{}],
      /*defaultImplementation=*/[{
        return cir::isExternalLinkage($_op.getLinkage());
      }]
      >,
      InterfaceMethod<"",
      "bool", "hasAvailableExternallyLinkage", (ins), [{}],
      /*defaultImplementation=*/[{
        return cir::isAvailableExternallyLinkage($_op.getLinkage());
      }]
      >,
      InterfaceMethod<"",
      "bool", "hasLinkOnceLinkage", (ins), [{}],
      /*defaultImplementation=*/[{
        return cir::isLinkOnceLinkage($_op.getLinkage());
      }]
      >,
      InterfaceMethod<"",
      "bool", "hasLinkOnceAnyLinkage", (ins), [{}],
      /*defaultImplementation=*/[{
        return cir::isLinkOnceAnyLinkage($_op.getLinkage());
      }]
      >,
      InterfaceMethod<"",
      "bool", "hasLinkOnceODRLinkage", (ins), [{}],
      /*defaultImplementation=*/[{
        return cir::isLinkOnceODRLinkage($_op.getLinkage());
      }]
      >,
      InterfaceMethod<"",
      "bool", "hasWeakLinkage", (ins), [{}],
      /*defaultImplementation=*/[{
        return cir::isWeakLinkage($_op.getLinkage());
      }]
      >,
      InterfaceMethod<"",
      "bool", "hasWeakAnyLinkage", (ins), [{}],
      /*defaultImplementation=*/[{
        return cir::isWeakAnyLinkage($_op.getLinkage());
      }]
      >,
      InterfaceMethod<"",
      "bool", "hasWeakODRLinkage", (ins), [{}],
      /*defaultImplementation=*/[{
        return cir::isWeakODRLinkage($_op.getLinkage());
      }]
      >,
      InterfaceMethod<"",
      "bool", "hasInternalLinkage", (ins), [{}],
      /*defaultImplementation=*/[{
        return cir::isInternalLinkage($_op.getLinkage());
      }]
      >,
      InterfaceMethod<"",
      "bool", "hasPrivateLinkage", (ins), [{}],
      /*defaultImplementation=*/[{
        return cir::isPrivateLinkage($_op.getLinkage());
      }]
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
      "bool", "hasCommonLinkage", (ins), [{}],
      /*defaultImplementation=*/[{
        return cir::isCommonLinkage($_op.getLinkage());
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
      "bool", "hasComdat", (ins), [{}],
      /*defaultImplementation=*/[{
        return $_op.getComdat();
      }]
      >,
      InterfaceMethod<"",
      "void", "setDSOLocal", (ins "bool":$val), [{}],
      /*defaultImplementation=*/[{
        $_op.setDsoLocal(val);
      }]
      >,
      InterfaceMethod<"",
      "bool", "isDSOLocal", (ins), [{}],
      /*defaultImplementation=*/[{
        return $_op.getDsoLocal();
      }]
      >,
      InterfaceMethod<"",
      "bool", "isWeakForLinker", (ins), [{}],
      /*defaultImplementation=*/[{
        return cir::isWeakForLinker($_op.getLinkage());
      }]
      >
    ];
    let extraClassDeclaration = [{
      bool hasDefaultVisibility();
      bool canBenefitFromLocalAlias();
    }];
  }

} // namespace cir

#endif // CLANG_CIR_INTERFACES_CIROPINTERFACES_TD
