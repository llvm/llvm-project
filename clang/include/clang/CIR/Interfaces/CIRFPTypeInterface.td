//===- CIRFPTypeInterface.td - CIR FP Interface Definitions -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CIR_INTERFACES_CIR_FP_TYPE_INTERFACE
#define MLIR_CIR_INTERFACES_CIR_FP_TYPE_INTERFACE

include "mlir/IR/OpBase.td"

def CIRFPTypeInterface : TypeInterface<"CIRFPTypeInterface"> {
  let description = [{
    Contains helper functions to query properties about a floating-point type.
  }];
  let cppNamespace = "::mlir::cir";

  let methods = [
    InterfaceMethod<[{
        Returns the bit width of this floating-point type.
      }],
      /*retTy=*/"unsigned",
      /*methodName=*/"getWidth",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
          return llvm::APFloat::semanticsSizeInBits($_type.getFloatSemantics());
        }]
    >,
    InterfaceMethod<[{
        Return the mantissa width.
      }],
      /*retTy=*/"unsigned",
      /*methodName=*/"getFPMantissaWidth",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
          return llvm::APFloat::semanticsPrecision($_type.getFloatSemantics());
        }]
    >,
    InterfaceMethod<[{
        Return the float semantics of this floating-point type.
      }],
      /*retTy=*/"const llvm::fltSemantics &",
      /*methodName=*/"getFloatSemantics"
    >,
  ];
}

#endif // MLIR_CIR_INTERFACES_CIR_FP_TYPE_INTERFACE
