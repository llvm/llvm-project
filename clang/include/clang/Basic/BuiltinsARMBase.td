//===--- BuiltinsARMBase.td - ARM Builtin function classes ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ARM-specific builtin function classes.
//
//===----------------------------------------------------------------------===//
include "clang/Basic/BuiltinsBase.td"

def ARMPrefix : NamePrefix<"__builtin_arm_">;

class ARMBuiltin<string prototype> : Builtin {
    let Spellings = [NAME];
    let Prototype = prototype;
    let RequiredNamePrefix = ARMPrefix;
}

class ARMNoPrefixBuiltin<string prototype> : Builtin {
    let Spellings = [NAME];
    let Prototype = prototype;
}

class ARMLangBuiltin<string prototype, string languages=""> : LangBuiltin<languages>  {
    let Spellings = [NAME];
    let Prototype = prototype;
    let RequiredNamePrefix = ARMPrefix;
}

class ARMNoPrefixLangBuiltin<string prototype, string languages=""> : LangBuiltin<languages> {
    let Spellings = [NAME];
    let Prototype = prototype;
}

class ARMTargetBuiltin<string prototype> : TargetBuiltin {
    let Spellings = [NAME];
    let Prototype = prototype;
    let RequiredNamePrefix = ARMPrefix;
}

class ARMNoPrefixTargetBuiltin<string prototype> : TargetBuiltin {
    let Spellings = [NAME];
    let Prototype = prototype;
}

class ARMNoPrefixTargetLibBuiltin<string prototype> : TargetLibBuiltin {
    let Spellings = [NAME];
    let Prototype = prototype;
}