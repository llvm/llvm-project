//===--- BuiltinsXCore.td - XCore Builtin function database ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "clang/Basic/BuiltinsBase.td"

class XCoreBuiltin<string prototype, list<Attribute> Attr = []> : TargetBuiltin {
  let Spellings = [NAME];
  let Prototype = prototype;
  let Attributes = Attr;
}

def __builtin_bitrev : XCoreBuiltin<"unsigned int(unsigned int)", [NoThrow, Const]>;
def __builtin_getid : XCoreBuiltin<"signed int()", [NoThrow, Const]>;
def __builtin_getps : XCoreBuiltin<"unsigned int(unsigned int)", [NoThrow]>;
def __builtin_setps : XCoreBuiltin<"void(unsigned int, unsigned int)", [NoThrow]>;
