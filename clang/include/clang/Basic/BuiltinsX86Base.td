//===--- BuiltinsX86Base.td - X86 Builtin function classes ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the X86-specific builtin function classes.
//
//===----------------------------------------------------------------------===//

include "clang/Basic/BuiltinsBase.td"

class X86Builtin<string prototype> : TargetBuiltin {
  let Spellings = ["__builtin_ia32_" # NAME];
  let Prototype = prototype;
  let EnableOpenCLLong = 1;
}

class X86NoPrefixBuiltin<string prototype> : TargetBuiltin {
  let Spellings = [NAME];
  let Prototype = prototype;
}

class X86LibBuiltin<string prototype> : TargetLibBuiltin {
  let Spellings = [NAME];
  let Prototype = prototype;
}
