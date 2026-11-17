//===-- BuiltinsZOS.td - z/OS Builtin function database ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the z/OS-specific builtin function database.
//
//===----------------------------------------------------------------------===//

include "clang/Basic/BuiltinsBase.td"

class SystemZNoPrefixBuiltin<string prototype> : Builtin {
  let Spellings = [NAME];
  let Prototype = prototype;
}

// compare and swap functions
let Attributes = [NoThrow] in {
  def __cs : SystemZNoPrefixBuiltin<"int(unsigned int *, unsigned int *, unsigned int)">;
  def __cs1 : SystemZNoPrefixBuiltin<"int(void *, void *, void *)">;
  def __csg : SystemZNoPrefixBuiltin<"int(void *, void *, void *)">;
  def __cds1 : SystemZNoPrefixBuiltin<"int(void *, void *, void *)">;
  def __cdsg : SystemZNoPrefixBuiltin<"int(void *, void *, void *)">;
}
