//===--- BuiltinsX86.td - X86 Builtin function database ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the X86-specific builtin function database.
//
//===----------------------------------------------------------------------===//

include "clang/Basic/BuiltinsBase.td"

class X86Builtin<string prototype> : TargetBuiltin {
  let Spellings = ["__builtin_ia32_" # NAME];
  let Prototype = prototype;
}

// Undefined Values
def undef128 : X86Builtin<"_Vector<2, double>()"> {
  let Attributes = [Const, NoThrow, RequiredVectorWidth<128>];
}

def undef256 : X86Builtin<"_Vector<4, double>()"> {
  let Attributes = [Const, NoThrow, RequiredVectorWidth<256>];
}

def undef512 : X86Builtin<"_Vector<8, double>()"> {
  let Attributes = [Const, NoThrow, RequiredVectorWidth<512>];
}

// FLAGS
def readeflags_u32 : X86Builtin<"unsigned int()"> {
  let Attributes = [NoThrow];
}

def writeeflags_u32 : X86Builtin<"void(unsigned int)"> {
  let Attributes = [NoThrow];
}

// MMX
//
// All MMX instructions will be generated via builtins. Any MMX vector
// types (<1 x i64>, <2 x i32>, etc.) that aren't used by these builtins will be
// expanded by the back-end.

def emms : X86Builtin<"void()"> {
  let Attributes = [NoThrow];
  let Features = "mmx";
}

let Attributes = [NoThrow, Const, RequiredVectorWidth<64>], Features = "sse" in {
  def vec_ext_v4hi : X86Builtin<"short(_Vector<4, short>, _Constant int)">;
  def vec_set_v4hi : X86Builtin<"_Vector<4, short>(_Vector<4, short>, short, _Constant int)">;
}

// SSE intrinsics
let Attributes = [Const, NoThrow, RequiredVectorWidth<128>] in {
  foreach Cmp = ["eq", "lt", "le", "gt", "ge", "neq"] in {
    let Features = "sse" in {
      def comi#Cmp : X86Builtin<"int(_Vector<4, float>, _Vector<4, float>)">;
      def ucomi#Cmp : X86Builtin<"int(_Vector<4, float>, _Vector<4, float>)">;
    }
    let Features = "sse2" in {
      def comisd#Cmp : X86Builtin<"int(_Vector<2, double>, _Vector<2, double>)">;
      def ucomisd#Cmp : X86Builtin<"int(_Vector<2, double>, _Vector<2, double>)">;
    }
  }

  foreach Cmp = ["cmpeq", "cmplt", "cmple", "cmpunord", "cmpneq", "cmpnlt",
                 "cmpnle", "cmpord", "min", "max"] in {
    let Features = "sse" in {
      def Cmp#ps : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>)">;
      def Cmp#ss : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>)">;
    }
    let Features = "sse2" in {
      def Cmp#pd : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>)">;
      def Cmp#sd : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>)">;
    }
  }

  let Features = "sse" in {
    def cmpps : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Constant char)">;
    def cmpss : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Constant char)">;
  }

  let Features = "sse2" in {
    def cmppd : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Constant char)">;
    def cmpsd : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Constant char)">;
  }

  let Features = "sse2" in {
    def pmulhw128 : X86Builtin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>)">;
    def pavgb128 : X86Builtin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>)">;
    def pavgw128 : X86Builtin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>)">;
    def packsswb128 : X86Builtin<"_Vector<16, char>(_Vector<8, short>, _Vector<8, short>)">;
    def packssdw128 : X86Builtin<"_Vector<8, short>(_Vector<4, int>, _Vector<4, int>)">;
    def packuswb128 : X86Builtin<"_Vector<16, char>(_Vector<8, short>, _Vector<8, short>)">;
    def pmulhuw128 : X86Builtin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>)">;
    def vec_ext_v2di : X86Builtin<"long long int(_Vector<2, long long int>, _Constant int)">;
    def vec_ext_v4si : X86Builtin<"int(_Vector<4, int>, _Constant int)">;
    def vec_ext_v4sf : X86Builtin<"float(_Vector<4, float>, _Constant int)">;
    def vec_ext_v8hi : X86Builtin<"short(_Vector<8, short>, _Constant int)">;
    def vec_set_v8hi : X86Builtin<"_Vector<8, short>(_Vector<8, short>, short, _Constant int)">;
  }

  let Features = "sse3" in {
    foreach Op = ["addsub", "hadd", "hsub"] in {
      def Op#ps : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>)">;
      def Op#pd : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>)">;
    }
  }

  let Features = "ssse3" in {
    foreach Op = ["phadd", "phsub"] in {
      def Op#w128 : X86Builtin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>)">;
      def Op#sw128 : X86Builtin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>)">;
      def Op#d128 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>)">;
    }

    def pmaddubsw128 : X86Builtin<"_Vector<8, short>(_Vector<16, char>, _Vector<16, char>)">;
    def pmulhrsw128 : X86Builtin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>)">;
    def pshufb128 : X86Builtin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>)">;
    def psignb128 : X86Builtin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>)">;
    def psignw128 : X86Builtin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>)">;
    def psignd128 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>)">;
  }
}

// AVX
let Attributes = [Const, NoThrow, RequiredVectorWidth<256>], Features = "avx" in {
  foreach Op = ["addsub", "hadd", "hsub", "max", "min"] in {
    def Op#pd256 : X86Builtin<"_Vector<4, double>(_Vector<4, double>, _Vector<4, double>)">;
    def Op#ps256 : X86Builtin<"_Vector<8, float>(_Vector<8, float>, _Vector<8, float>)">;
  }
}
