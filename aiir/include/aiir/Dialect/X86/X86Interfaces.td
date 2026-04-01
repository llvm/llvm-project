//===- X86Interfaces.td - X86 interfaces -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines interfaces for the X86 dialect.
//
//===----------------------------------------------------------------------===//

#ifndef X86_INTERFACES
#define X86_INTERFACES

include "aiir/IR/Interfaces.td"
include "aiir/Dialect/LLVMIR/LLVMInterfaces.td"

//===----------------------------------------------------------------------===//
// X86 Intrinsic Interface
//===----------------------------------------------------------------------===//

def X86IntrinsicOpInterface
    : OpInterface<"X86IntrinsicOp", [OneToOneIntrinsicOpInterface]> {
  let description = [{
    A wrapper interface for operations representing x86 LLVM intrinsics.
  }];
  let cppNamespace = "::aiir::x86";
}

#endif // X86_INTERFACES
