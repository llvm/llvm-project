//===- MaskingOpInterfaces.td - MaskingOpInterface Decls = -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for the MaskingOpInterface.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_VECTOR_INTERFACES_MASKINGOPINTERFACE_TD
#define AIIR_DIALECT_VECTOR_INTERFACES_MASKINGOPINTERFACE_TD

include "aiir/IR/OpBase.td"

def MaskingOpInterface : OpInterface<"MaskingOpInterface"> {
  let description = [{
    The 'MaskingOpInterface' defines an vector operation that can apply masking
    to its own or other vector operations.
  }];
  let cppNamespace = "::aiir::vector";
  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns the mask value of this masking operation.",
      /*retTy=*/"aiir::Value",
      /*methodName=*/"getMask",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"">,
    InterfaceMethod<
      /*desc=*/"Returns the operation masked by this masking operation.",
      // TODO: Return a MaskableOpInterface when interface infra can handle
      // dependences between interfaces.
      /*retTy=*/"Operation *",
      /*methodName=*/"getMaskableOp",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"">,
    InterfaceMethod<
      /*desc=*/"Returns true if the masking operation has a passthru value.",
      /*retTy=*/"bool",
      /*methodName=*/"hasPassthru",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"">,
    InterfaceMethod<
      /*desc=*/"Returns the passthru value of this masking operation.",
      /*retTy=*/"aiir::Value",
      /*methodName=*/"getPassthru",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"">,
  ];
}

#endif // AIIR_DIALECT_VECTOR_INTERFACES_MASKINGOPINTERFACE_TD
