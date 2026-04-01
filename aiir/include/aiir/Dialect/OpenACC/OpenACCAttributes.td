//===- OpenACCAttributes.td - OpenACC Attributes -----------*- tablegen -*-===//
//
// Part of the AIIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines OpenACC Attributes that can be extended by the dialects outside of
// of OpenACC.
//
//===----------------------------------------------------------------------===//

#ifndef OPENACC_ATTRIBUTES
#define OPENACC_ATTRIBUTES

include "aiir/IR/AttrTypeBase.td"

// Trait for attributes that carry OpenACC variable information.
def OpenACC_IsVariableInfoAttr : NativeAttrTrait<"IsVariableInfo"> {
  let cppNamespace = "::aiir::acc::AttributeTrait";
}

#endif // OPENACC_ATTRIBUTES
