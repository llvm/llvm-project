//===--- BuiltinsSPIRVVK.td - SPIRV Builtin function database ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "clang/Basic/BuiltinsSPIRVBase.td"


def reflect : SPIRVBuiltin<"void(...)", [NoThrow, Const]>;
def faceforward : SPIRVBuiltin<"void(...)", [NoThrow, Const, CustomTypeChecking]>;
def refract : SPIRVBuiltin<"void(...)", [NoThrow, Const, CustomTypeChecking]>;
def fwidth : SPIRVBuiltin<"void(...)", [NoThrow, Const, CustomTypeChecking]>;
