//===--- BuiltinsAArch64Base.td - AArch64 Builtin function classes ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the AArch64-specific builtin function classes.
//
//===----------------------------------------------------------------------===//

include "clang/Basic/BuiltinsBase.td"

def AArch64Prefix : NamePrefix<"__builtin_arm_">;

class AArch64Builtin<string prototype> : Builtin {
  let Spellings = [NAME];
  let Prototype = prototype;
  let RequiredNamePrefix = AArch64Prefix;
}

class AArch64NoPrefixBuiltin<string prototype> : Builtin {
  let Spellings = [NAME];
  let Prototype = prototype;
}

class AArch64TargetBuiltin<string prototype> : TargetBuiltin {
  let Spellings = [NAME];
  let Prototype = prototype;
  let RequiredNamePrefix = AArch64Prefix;
}

class AArch64NoPrefixTargetLibBuiltin<string prototype> : TargetLibBuiltin {
  let Spellings = [NAME];
  let Prototype = prototype;
}
