//===--- BuiltinsSPIRV.td - SPIRV Builtin function database ---------*- C++ -*-===//
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

def SPIRVDistance : Builtin {
  let Spellings = ["__builtin_spirv_distance"];
  let Attributes = [NoThrow, Const];
  let Prototype = "void(...)";
}

def SPIRVLength : Builtin {
  let Spellings = ["__builtin_spirv_length"];
  let Attributes = [NoThrow, Const];
  let Prototype = "void(...)";
}

def SPIRVReflect : Builtin {
  let Spellings = ["__builtin_spirv_reflect"];
  let Attributes = [NoThrow, Const];
  let Prototype = "void(...)";
}

def SPIRVSmoothStep : Builtin {
  let Spellings = ["__builtin_spirv_smoothstep"];
  let Attributes = [NoThrow, Const, CustomTypeChecking];
  let Prototype = "void(...)";
}

def SPIRVFaceForward : Builtin {
  let Spellings = ["__builtin_spirv_faceforward"];
  let Attributes = [NoThrow, Const, CustomTypeChecking];
  let Prototype = "void(...)";
}

def generic_cast_to_ptr_explicit
    : SPIRVBuiltin<"void*(void*, int)", [NoThrow, Const, CustomTypeChecking]>;
