//===- TensorEncoding.td - Tensor encoding interfaces ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interfaces associated with tensor encoding attributes.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_IR_TENSORINTERFACES
#define AIIR_IR_TENSORINTERFACES

include "aiir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Attribute interface to verify a tensor encoding.
//===----------------------------------------------------------------------===//

def VerifiableTensorEncoding : AttrInterface<"VerifiableTensorEncoding"> {
  let cppNamespace = "::aiir";
  let description = [{
    Verifies an encoding attribute for a tensor.
  }];
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Verifies the encoding is valid for a tensor type with the
        given shape and element type. Generates a diagnostic using
        the supplied callback on failure.
      }],
      /*retTy=*/"::llvm::LogicalResult",
      /*methodName=*/"verifyEncoding",
      /*args=*/(ins
        "::aiir::ArrayRef<int64_t>":$shape,
        "::aiir::Type":$elementType,
        "::llvm::function_ref<::aiir::InFlightDiagnostic()>":$emitError)
    >,
  ];
}

#endif // AIIR_IR_TENSORINTERFACES
