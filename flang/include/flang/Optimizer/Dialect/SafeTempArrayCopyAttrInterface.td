//===- SafeTempArrayCopyAttrInterface.td -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file defines SafeTempArrayCopyAttrInterface and a generic attribute
/// SafeTempArrayCopyAttr promising the SafeTempArrayCopyAttrInterface.
///
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SAFETEMPARRAYCOPYATTRINTERFACE_TD
#define FORTRAN_SAFETEMPARRAYCOPYATTRINTERFACE_TD

include "mlir/IR/OpBase.td"

def SafeTempArrayCopyAttrInterface
    : AttrInterface<"SafeTempArrayCopyAttrInterface"> {
  let cppNamespace = "::fir";
  let description = [{
    Interface for attributes defining whether creation of a temporary
    copy of a Fortran array is safe and/or how to produce proper
    dynamic checks to avoid it, if it is unsafe.
  }];

  let methods =
      [StaticInterfaceMethod<
           /*desc=*/[{
        Returns true iff the usage of the temporary array copy
        can be made safe applying some dynamic checks.
      }],
           /*retTy=*/"bool",
           /*methodName=*/"isDynamicallySafe",
           /*args=*/(ins)>,
       StaticInterfaceMethod<
           /*desc=*/[{
        Generate FIR that produces an i1 Value indicating
        whether the creation of the temporary array copy is safe.
        \p array is a definition of the original array.
        The implementation may assume that \p array is present
        (though, it may be empty).
      }],
           /*retTy=*/"mlir::Value",
           /*methodName=*/"genDynamicCheck",
           /*args=*/
           (ins "::mlir::Location":$loc, "::fir::FirOpBuilder &":$builder,
               "::mlir::Value":$array)>,
       StaticInterfaceMethod<
           /*desc=*/[{
        This method allows inserting any FIR right before the optional
        copy-out (from \p temp to \p array) and the deallocation
        of the temporary array (implying that the temporary copy was
        actually created).
      }],
           /*retTy=*/"void",
           /*methodName=*/"registerTempDeallocation",
           /*args=*/
           (ins "::mlir::Location":$loc, "::fir::FirOpBuilder &":$builder,
               "::mlir::Value":$array, "::mlir::Value":$temp)>,
  ];
}

def SafeTempArrayCopyAttr
    : ConfinedAttr<
          AnyAttr, [PromisedAttrInterface<SafeTempArrayCopyAttrInterface>]> {
  let description = [{
    Generic attribute implementing or promising
    the `SafeTempArrayCopyAttrInterface` interface.
  }];
}

def SafeTempArrayCopyArrayAttr
    : TypedArrayAttrBase<SafeTempArrayCopyAttr,
                         "array of SafeTempArrayCopyAttr attributes">;

def NonEmptySafeTempArrayCopyArrayAttr
    : ConfinedAttr<SafeTempArrayCopyArrayAttr, [ArrayMinCount<1>]>;

#endif // FORTRAN_SAFETEMPARRAYCOPYATTRINTERFACE_TD
