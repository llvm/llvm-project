//===-- FIROperationMoveOpInterface.td ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "mlir/IR/Interfaces.td"

def OperationMoveOpInterface : OpInterface<"OperationMoveOpInterface"> {
  let description = [{
    Provides methods to control movement of operations during transformations.
    For example, some LoopLikeOpInterface operations may disallow
    hoisting speculatable operations (e.g. even Pure ones) under some
    conditions. Operations may control when the hoisting
    (or rather a general movement) is allowed by implementing this interface.
  }];
  let cppNamespace = "::fir";
  let verify = [{ return detail::verifyOperationMoveOpInterface($_op); }];
  let methods =
      [InterfaceMethod<
           /*desc=*/[{
        Returns true if it is allowed to move the given 'candidate'
        operation from the 'descendant' operation into this operation.
        If 'candidate' is nullptr, then the caller is querying whether
        any operation from any descendant can be moved into this operation.

        The implementations of OperationMoveOpInterface should not provide
        implementation of this method, in general. The default implementation
        applies some verification and relays to canMoveFromDescendantImpl()
        method, which the implementations must provide.
      }],
           /*returnType=*/"bool",
           /*methodName=*/"canMoveFromDescendant",
           /*args=*/
           (ins "::mlir::Operation *":$descendant,
               "::mlir::Operation *":$candidate),
           /*methodBody=*/[{}],
           /*defaultImplementation=*/[{
        return detail::canMoveFromDescendant($_op, descendant, candidate);
      }]>,
       InterfaceMethod<
           /*desc=*/"",
           /*returnType=*/"bool",
           /*methodName=*/"canMoveFromDescendantImpl",
           /*args=*/
           (ins "::mlir::Operation *":$descendant,
               "::mlir::Operation *":$candidate)>,
       InterfaceMethod<
           /*desc=*/[{
        Returns true if it is allowed to move the given 'candidate'
        operation out of this operation. If 'candidate' is nullptr,
        then the caller is querying whether any operation can be moved
        out of this operation.

        The implementations of OperationMoveOpInterface should not provide
        implementation of this method, in general. The default implementation
        applies some verification and relays to canMoveOutOfImpl() method,
        which the implementations must provide.
      }],
           /*returnType=*/"bool",
           /*methodName=*/"canMoveOutOf",
           /*args=*/(ins "::mlir::Operation *":$candidate),
           /*methodBody=*/[{}],
           /*defaultImplementation=*/[{
        return detail::canMoveOutOf($_op, candidate);
      }]>,
       InterfaceMethod<
           /*desc=*/"Implementation of canMoveOutOf() for concrete operation",
           /*returnType=*/"bool",
           /*methodName=*/"canMoveOutOfImpl",
           /*args=*/(ins "::mlir::Operation *":$candidate)>,
  ];
}
