//===--- BuiltinsAVR.td - AVR Builtin function database ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "clang/Basic/BuiltinsBase.td"

def AVR_NOP : TargetBuiltin {
  let Spellings = ["__builtin_avr_nop"];
  let Attributes = [NoThrow];
  let Prototype = "void()";
}

def AVR_SEI : TargetBuiltin {
  let Spellings = ["__builtin_avr_sei"];
  let Attributes = [NoThrow];
  let Prototype = "void()";
}

def AVR_CLI : TargetBuiltin {
  let Spellings = ["__builtin_avr_cli"];
  let Attributes = [NoThrow];
  let Prototype = "void()";
}

def AVR_SLEEP : TargetBuiltin {
  let Spellings = ["__builtin_avr_sleep"];
  let Attributes = [NoThrow];
  let Prototype = "void()";
}

def AVR_WDR : TargetBuiltin {
  let Spellings = ["__builtin_avr_wdr"];
  let Attributes = [NoThrow];
  let Prototype = "void()";
}

def AVR_SWAP : TargetBuiltin {
  let Spellings = ["__builtin_avr_swap"];
  let Attributes = [NoThrow, Const];
  let Prototype = "unsigned char(unsigned char)";
}