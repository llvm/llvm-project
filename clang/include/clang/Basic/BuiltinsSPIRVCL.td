//===--- BuiltinsSPIRVCL.td - SPIRV Builtin function database ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "clang/Basic/BuiltinsSPIRVBase.td"

def generic_cast_to_ptr_explicit
    : SPIRVBuiltin<"void*(void*, int)", [NoThrow, Const, CustomTypeChecking]>;
