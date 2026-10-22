//===--- BuiltinsMeta.td - Builtins for reflection metafunctions ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------------===//

// All metafunctions should be consteval.
class CXXMetaBuiltin<string name, string prototype, list<Attribute> Attrs = []>
    : LangBuiltin<"CXX_LANG"> {
  let Spellings = [!strconcat("__builtin_meta_", name)];
  let Prototype = prototype;
  let Attributes = !listconcat([Consteval], Attrs);
}

def MetaIsType : CXXMetaBuiltin<"is_type", "bool(std::meta::info)", [NoThrow]>;
