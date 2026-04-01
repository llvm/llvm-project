//===-- OpenACCMPOpsInterfaces.td - OpenACC/MP op interfaces - tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the OpenACC/OpenMP Dialect interfaces definition file.
//
//===----------------------------------------------------------------------===//

#ifndef OPENACC_MP_COMMON_OPS_INTERFACES
#define OPENACC_MP_COMMON_OPS_INTERFACES

include "aiir/IR/OpBase.td"

def RecipeInterface : OpInterface<"RecipeInterface"> {
    let description = [{
      OpenACC/OpenMP operations with one or more regions holding executable code.
    }];
    let cppNamespace = "::aiir::accomp";
    let methods = [
      InterfaceMethod<
        /*description=*/[{
          For the given region of the operation return the block
          inside the region, where an alloca-like operation should be inserted.
          The default implementation returns the entry block of the region.
        }],
        /*retTy*/"::aiir::Block *",
        /*methodName=*/"getAllocaBlock",
        /*args=*/(ins "::aiir::Region &":$region),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return &region.front();
        }]
      >,
    ];
}

#endif // OPENACC_MP_COMMON_OPS_INTERFACES
