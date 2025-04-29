//===- BuiltinsSPIRVCommon.td - SPIRV Builtin function database -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "clang/Basic/BuiltinsSPIRVBase.td"

def distance : SPIRVBuiltin<"void(...)", [NoThrow, Const]>;
def length : SPIRVBuiltin<"void(...)", [NoThrow, Const]>;
def smoothstep : SPIRVBuiltin<"void(...)", [NoThrow, Const, CustomTypeChecking]>;
