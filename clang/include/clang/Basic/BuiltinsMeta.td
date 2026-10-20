//===--- BuiltinsMeta.td - Builtins for reflection metafunctions ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------------===//

// All metafunctions should be consteval. 
class CXXMetaBuiltin<string prototype, list<Attribute> Attrs = []>
    : LangBuiltin<"CXX_LANG"> {
  let Prototype = prototype;
  let Attributes = !listconcat([Consteval], Attrs);
}

def MetaIsType : CXXMetaBuiltin<"bool(std::meta::info)", [NoThrow]> {
  let Spellings = ["__builtin_meta_is_type"];
}
