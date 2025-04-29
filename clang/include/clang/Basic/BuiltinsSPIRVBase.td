//===--- BuiltinsSPIRVBase.td - SPIRV Builtin function database -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "clang/Basic/BuiltinsBase.td"

class SPIRVBuiltin<string prototype, list<Attribute> Attr> : Builtin {
  let Spellings = ["__builtin_spirv_"#NAME];
  let Prototype = prototype;
  let Attributes = !listconcat([NoThrow], Attr);
}
