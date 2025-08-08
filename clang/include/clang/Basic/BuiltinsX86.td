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

include "clang/Basic/BuiltinsX86Base.td"

def rdpmc : X86Builtin<"unsigned long long int(int)">;
def rdtsc : X86Builtin<"unsigned long long int()">;
def __rdtsc : X86NoPrefixBuiltin<"unsigned long long int()"> {
  let EnableOpenCLLong = 1;
}
def rdtscp : X86Builtin<"unsigned long long int(unsigned int*)">;

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
    def pavgb128 : X86Builtin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>)">;
    def pavgw128 : X86Builtin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>)">;
    def packsswb128 : X86Builtin<"_Vector<16, char>(_Vector<8, short>, _Vector<8, short>)">;
    def packssdw128 : X86Builtin<"_Vector<8, short>(_Vector<4, int>, _Vector<4, int>)">;
    def packuswb128 : X86Builtin<"_Vector<16, char>(_Vector<8, short>, _Vector<8, short>)">;
    def vec_ext_v2di : X86Builtin<"long long int(_Vector<2, long long int>, _Constant int)">;
    def vec_ext_v4si : X86Builtin<"int(_Vector<4, int>, _Constant int)">;
    def vec_ext_v4sf : X86Builtin<"float(_Vector<4, float>, _Constant int)">;
    def vec_ext_v8hi : X86Builtin<"short(_Vector<8, short>, _Constant int)">;
    def vec_set_v8hi : X86Builtin<"_Vector<8, short>(_Vector<8, short>, short, _Constant int)">;
  }

  let Features = "sse2", Attributes = [NoThrow, Const, Constexpr, RequiredVectorWidth<128>] in {
    def pmulhw128 : X86Builtin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>)">;
    def pmulhuw128 : X86Builtin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)">;
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

let Features = "mmx", Header = "mmintrin.h", Attributes = [NoThrow, Const] in {
  def _m_prefetch : X86LibBuiltin<"void(void *)">;
}

// PRFCHW
let Features = "prfchw", Header = "intrin.h", Attributes = [NoThrow, Const] in {
  def _m_prefetchw : X86LibBuiltin<"void(void volatile const *)">;
}

// Mechanically ported builtins from the original `.def` file.
//
// TODO: Build structured ways of synthesizing relevant groups and improve the
// organization of the builtins below this point (and move them above it). The
// current formulation is based on what was easiest to recognize from the
// pre-TableGen version.

let Features = "sse", Header = "xmmintrin.h", Attributes = [NoThrow, Const] in {
  def _mm_prefetch : X86LibBuiltin<"void(char const *, int)">;
}

let Features = "sse", Attributes = [NoThrow] in {
  def ldmxcsr : X86Builtin<"void(unsigned int)">;
}

let Features = "sse", Header = "xmmintrin.h", Attributes = [NoThrow, RequireDeclaration] in {
  def _mm_setcsr : X86LibBuiltin<"void(unsigned int)">;
}

let Features = "sse", Attributes = [NoThrow] in {
  def stmxcsr : X86Builtin<"unsigned int()">;
}

let Features = "sse", Header = "xmmintrin.h", Attributes = [NoThrow, RequireDeclaration] in {
  def _mm_getcsr : X86LibBuiltin<"unsigned int()">;
}

let Features = "sse", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def cvtss2si : X86Builtin<"int(_Vector<4, float>)">;
  def cvttss2si : X86Builtin<"int(_Vector<4, float>)">;
}

let Features = "sse", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def movmskps : X86Builtin<"int(_Vector<4, float>)">;
}

let Features = "sse", Attributes = [NoThrow] in {
  def sfence : X86Builtin<"void()">;
}

let Features = "sse", Header = "xmmintrin.h", Attributes = [NoThrow, RequireDeclaration] in {
  def _mm_sfence : X86LibBuiltin<"void()">;
}

let Features = "sse", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def rcpps : X86Builtin<"_Vector<4, float>(_Vector<4, float>)">;
  def rcpss : X86Builtin<"_Vector<4, float>(_Vector<4, float>)">;
  def rsqrtps : X86Builtin<"_Vector<4, float>(_Vector<4, float>)">;
  def rsqrtss : X86Builtin<"_Vector<4, float>(_Vector<4, float>)">;
  def sqrtps : X86Builtin<"_Vector<4, float>(_Vector<4, float>)">;
  def sqrtss : X86Builtin<"_Vector<4, float>(_Vector<4, float>)">;
  def shufps : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Constant int)">;
}

let Features = "sse2", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def maskmovdqu : X86Builtin<"void(_Vector<16, char>, _Vector<16, char>, char *)">;
}

let Features = "sse2", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def movmskpd : X86Builtin<"int(_Vector<2, double>)">;
  def pmovmskb128 : X86Builtin<"int(_Vector<16, char>)">;
}

let Features = "sse2", Attributes = [NoThrow] in {
  def movnti : X86Builtin<"void(int *, int)">;
}

let Features = "sse2", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def pshufd : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Constant int)">;
  def pshuflw : X86Builtin<"_Vector<8, short>(_Vector<8, short>, _Constant int)">;
  def pshufhw : X86Builtin<"_Vector<8, short>(_Vector<8, short>, _Constant int)">;
  def psadbw128 : X86Builtin<"_Vector<2, long long int>(_Vector<16, char>, _Vector<16, char>)">;
  def sqrtpd : X86Builtin<"_Vector<2, double>(_Vector<2, double>)">;
  def sqrtsd : X86Builtin<"_Vector<2, double>(_Vector<2, double>)">;
  def shufpd : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Constant int)">;
  def cvtpd2dq : X86Builtin<"_Vector<2, long long int>(_Vector<2, double>)">;
  def cvtpd2ps : X86Builtin<"_Vector<4, float>(_Vector<2, double>)">;
  def cvttpd2dq : X86Builtin<"_Vector<4, int>(_Vector<2, double>)">;
  def cvtsd2si : X86Builtin<"int(_Vector<2, double>)">;
  def cvttsd2si : X86Builtin<"int(_Vector<2, double>)">;
  def cvtsd2ss : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<2, double>)">;
  def cvtps2dq : X86Builtin<"_Vector<4, int>(_Vector<4, float>)">;
  def cvttps2dq : X86Builtin<"_Vector<4, int>(_Vector<4, float>)">;
}

let Features = "sse2", Attributes = [NoThrow] in {
  def clflush : X86Builtin<"void(void const *)">;
}

let Features = "sse2", Header = "emmintrin.h", Attributes = [NoThrow, RequireDeclaration] in {
  def _mm_clflush : X86LibBuiltin<"void(void const *)">;
}

let Features = "sse2", Attributes = [NoThrow] in {
  def lfence : X86Builtin<"void()">;
}

let Features = "sse2", Header = "emmintrin.h", Attributes = [NoThrow, RequireDeclaration] in {
  def _mm_lfence : X86LibBuiltin<"void()">;
}

let Features = "sse2", Attributes = [NoThrow] in {
  def mfence : X86Builtin<"void()">;
}

let Features = "sse2", Header = "emmintrin.h", Attributes = [NoThrow, RequireDeclaration] in {
  def _mm_mfence : X86LibBuiltin<"void()">;
}

let Attributes = [NoThrow] in {
  def pause : X86Builtin<"void()">;
}

let Header = "emmintrin.h", Attributes = [NoThrow, RequireDeclaration] in {
  def _mm_pause : X86LibBuiltin<"void()">;
}

let Features = "sse2", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def pmuludq128 : X86Builtin<"_Vector<2, long long int>(_Vector<4, int>, _Vector<4, int>)">;
  def psraw128 : X86Builtin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>)">;
  def psrad128 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>)">;
  def psrlw128 : X86Builtin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>)">;
  def psrld128 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>)">;
  def psrlq128 : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, _Vector<2, long long int>)">;
  def psllw128 : X86Builtin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>)">;
  def pslld128 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>)">;
  def psllq128 : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, _Vector<2, long long int>)">;
  def psllwi128 : X86Builtin<"_Vector<8, short>(_Vector<8, short>, int)">;
  def pslldi128 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, int)">;
  def psllqi128 : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, int)">;
  def psrlwi128 : X86Builtin<"_Vector<8, short>(_Vector<8, short>, int)">;
  def psrldi128 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, int)">;
  def psrlqi128 : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, int)">;
  def psrawi128 : X86Builtin<"_Vector<8, short>(_Vector<8, short>, int)">;
  def psradi128 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, int)">;
  def pmaddwd128 : X86Builtin<"_Vector<4, int>(_Vector<8, short>, _Vector<8, short>)">;
  def pslldqi128_byteshift : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, _Constant int)">;
  def psrldqi128_byteshift : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, _Constant int)">;
}

let Features = "sse3", Attributes = [NoThrow] in {
  def monitor : X86Builtin<"void(void const *, unsigned int, unsigned int)">;
  def mwait : X86Builtin<"void(unsigned int, unsigned int)">;
}

let Features = "sse3", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def lddqu : X86Builtin<"_Vector<16, char>(char const *)">;
}

let Features = "ssse3", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def palignr128 : X86Builtin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>, _Constant int)">;
}

let Features = "sse4.1", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def insertps128 : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Constant char)">;
  def pblendvb128 : X86Builtin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>)">;
  def pblendw128 : X86Builtin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>, _Constant int)">;
  def blendpd : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Constant int)">;
  def blendps : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Constant int)">;
  def blendvpd : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Vector<2, double>)">;
  def blendvps : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>)">;
  def packusdw128 : X86Builtin<"_Vector<8, short>(_Vector<4, int>, _Vector<4, int>)">;
  def pmuldq128 : X86Builtin<"_Vector<2, long long int>(_Vector<4, int>, _Vector<4, int>)">;
  def roundps : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Constant int)">;
  def roundss : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Constant int)">;
  def roundsd : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Constant int)">;
  def roundpd : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Constant int)">;
  def dpps : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Constant char)">;
  def dppd : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Constant char)">;
  def ptestz128 : X86Builtin<"int(_Vector<2, long long int>, _Vector<2, long long int>)">;
  def ptestc128 : X86Builtin<"int(_Vector<2, long long int>, _Vector<2, long long int>)">;
  def ptestnzc128 : X86Builtin<"int(_Vector<2, long long int>, _Vector<2, long long int>)">;
  def mpsadbw128 : X86Builtin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>, _Constant char)">;
  def phminposuw128 : X86Builtin<"_Vector<8, short>(_Vector<8, short>)">;
  def vec_ext_v16qi : X86Builtin<"char(_Vector<16, char>, _Constant int)">;
  def vec_set_v16qi : X86Builtin<"_Vector<16, char>(_Vector<16, char>, char, _Constant int)">;
  def vec_set_v4si : X86Builtin<"_Vector<4, int>(_Vector<4, int>, int, _Constant int)">;
}

let Features = "sse4.2", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def pcmpistrm128 : X86Builtin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>, _Constant char)">;
  def pcmpistri128 : X86Builtin<"int(_Vector<16, char>, _Vector<16, char>, _Constant char)">;
  def pcmpestrm128 : X86Builtin<"_Vector<16, char>(_Vector<16, char>, int, _Vector<16, char>, int, _Constant char)">;
  def pcmpestri128 : X86Builtin<"int(_Vector<16, char>, int, _Vector<16, char>, int, _Constant char)">;
  def pcmpistria128 : X86Builtin<"int(_Vector<16, char>, _Vector<16, char>, _Constant char)">;
  def pcmpistric128 : X86Builtin<"int(_Vector<16, char>, _Vector<16, char>, _Constant char)">;
  def pcmpistrio128 : X86Builtin<"int(_Vector<16, char>, _Vector<16, char>, _Constant char)">;
  def pcmpistris128 : X86Builtin<"int(_Vector<16, char>, _Vector<16, char>, _Constant char)">;
  def pcmpistriz128 : X86Builtin<"int(_Vector<16, char>, _Vector<16, char>, _Constant char)">;
  def pcmpestria128 : X86Builtin<"int(_Vector<16, char>, int, _Vector<16, char>, int, _Constant char)">;
  def pcmpestric128 : X86Builtin<"int(_Vector<16, char>, int, _Vector<16, char>, int, _Constant char)">;
  def pcmpestrio128 : X86Builtin<"int(_Vector<16, char>, int, _Vector<16, char>, int, _Constant char)">;
  def pcmpestris128 : X86Builtin<"int(_Vector<16, char>, int, _Vector<16, char>, int, _Constant char)">;
  def pcmpestriz128 : X86Builtin<"int(_Vector<16, char>, int, _Vector<16, char>, int, _Constant char)">;
}

let Features = "crc32", Attributes = [NoThrow, Const] in {
  def crc32qi : X86Builtin<"unsigned int(unsigned int, unsigned char)">;
  def crc32hi : X86Builtin<"unsigned int(unsigned int, unsigned short)">;
  def crc32si : X86Builtin<"unsigned int(unsigned int, unsigned int)">;
}

let Features = "sse4a", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def extrqi : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, _Constant char, _Constant char)">;
  def extrq : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, _Vector<16, char>)">;
  def insertqi : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, _Vector<2, long long int>, _Constant char, _Constant char)">;
  def insertq : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, _Vector<2, long long int>)">;
}

let Features = "sse4a", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def movntsd : X86Builtin<"void(double *, _Vector<2, double>)">;
  def movntss : X86Builtin<"void(float *, _Vector<4, float>)">;
}

let Features = "aes", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def aesenc128 : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, _Vector<2, long long int>)">;
  def aesenclast128 : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, _Vector<2, long long int>)">;
  def aesdec128 : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, _Vector<2, long long int>)">;
  def aesdeclast128 : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, _Vector<2, long long int>)">;
  def aesimc128 : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>)">;
  def aeskeygenassist128 : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, _Constant char)">;
}

let Features = "vaes", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def aesenc256 : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, _Vector<4, long long int>)">;
}

let Features = "avx512f,evex512,vaes", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def aesenc512 : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int>, _Vector<8, long long int>)">;
}

let Features = "vaes", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def aesenclast256 : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, _Vector<4, long long int>)">;
}

let Features = "avx512f,evex512,vaes", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def aesenclast512 : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int>, _Vector<8, long long int>)">;
}

let Features = "vaes", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def aesdec256 : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, _Vector<4, long long int>)">;
}

let Features = "avx512f,evex512,vaes", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def aesdec512 : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int>, _Vector<8, long long int>)">;
}

let Features = "vaes", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def aesdeclast256 : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, _Vector<4, long long int>)">;
}

let Features = "avx512f,evex512,vaes", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def aesdeclast512 : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int>, _Vector<8, long long int>)">;
}

let Features = "gfni", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vgf2p8affineinvqb_v16qi : X86Builtin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>, _Constant char)">;
}

let Features = "avx,gfni", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vgf2p8affineinvqb_v32qi : X86Builtin<"_Vector<32, char>(_Vector<32, char>, _Vector<32, char>, _Constant char)">;
}

let Features = "avx512f,evex512,gfni", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vgf2p8affineinvqb_v64qi : X86Builtin<"_Vector<64, char>(_Vector<64, char>, _Vector<64, char>, _Constant char)">;
}

let Features = "gfni", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vgf2p8affineqb_v16qi : X86Builtin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>, _Constant char)">;
}

let Features = "avx,gfni", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vgf2p8affineqb_v32qi : X86Builtin<"_Vector<32, char>(_Vector<32, char>, _Vector<32, char>, _Constant char)">;
}

let Features = "avx512f,evex512,gfni", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vgf2p8affineqb_v64qi : X86Builtin<"_Vector<64, char>(_Vector<64, char>, _Vector<64, char>, _Constant char)">;
}

let Features = "gfni", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vgf2p8mulb_v16qi : X86Builtin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>)">;
}

let Features = "avx,gfni", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vgf2p8mulb_v32qi : X86Builtin<"_Vector<32, char>(_Vector<32, char>, _Vector<32, char>)">;
}

let Features = "avx512f,evex512,gfni", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vgf2p8mulb_v64qi : X86Builtin<"_Vector<64, char>(_Vector<64, char>, _Vector<64, char>)">;
}

let Features = "pclmul", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def pclmulqdq128 : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, _Vector<2, long long int>, _Constant char)">;
}

let Features = "vpclmulqdq", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def pclmulqdq256 : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, _Vector<4, long long int>, _Constant char)">;
}

let Features = "avx512f,evex512,vpclmulqdq", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def pclmulqdq512 : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int>, _Vector<8, long long int>, _Constant char)">;
}

let Features = "avx", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vpermilvarpd : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, long long int>)">;
  def vpermilvarps : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, int>)">;
  def vpermilvarpd256 : X86Builtin<"_Vector<4, double>(_Vector<4, double>, _Vector<4, long long int>)">;
  def vpermilvarps256 : X86Builtin<"_Vector<8, float>(_Vector<8, float>, _Vector<8, int>)">;
  def blendpd256 : X86Builtin<"_Vector<4, double>(_Vector<4, double>, _Vector<4, double>, _Constant int)">;
  def blendps256 : X86Builtin<"_Vector<8, float>(_Vector<8, float>, _Vector<8, float>, _Constant int)">;
  def blendvpd256 : X86Builtin<"_Vector<4, double>(_Vector<4, double>, _Vector<4, double>, _Vector<4, double>)">;
  def blendvps256 : X86Builtin<"_Vector<8, float>(_Vector<8, float>, _Vector<8, float>, _Vector<8, float>)">;
  def shufpd256 : X86Builtin<"_Vector<4, double>(_Vector<4, double>, _Vector<4, double>, _Constant int)">;
  def shufps256 : X86Builtin<"_Vector<8, float>(_Vector<8, float>, _Vector<8, float>, _Constant int)">;
  def dpps256 : X86Builtin<"_Vector<8, float>(_Vector<8, float>, _Vector<8, float>, _Constant char)">;
  def cmppd256 : X86Builtin<"_Vector<4, double>(_Vector<4, double>, _Vector<4, double>, _Constant char)">;
  def cmpps256 : X86Builtin<"_Vector<8, float>(_Vector<8, float>, _Vector<8, float>, _Constant char)">;
  def vextractf128_pd256 : X86Builtin<"_Vector<2, double>(_Vector<4, double>, _Constant int)">;
  def vextractf128_ps256 : X86Builtin<"_Vector<4, float>(_Vector<8, float>, _Constant int)">;
  def vextractf128_si256 : X86Builtin<"_Vector<4, int>(_Vector<8, int>, _Constant int)">;
  def cvtpd2ps256 : X86Builtin<"_Vector<4, float>(_Vector<4, double>)">;
  def cvtps2dq256 : X86Builtin<"_Vector<8, int>(_Vector<8, float>)">;
  def cvttpd2dq256 : X86Builtin<"_Vector<4, int>(_Vector<4, double>)">;
  def cvtpd2dq256 : X86Builtin<"_Vector<4, int>(_Vector<4, double>)">;
  def cvttps2dq256 : X86Builtin<"_Vector<8, int>(_Vector<8, float>)">;
  def vperm2f128_pd256 : X86Builtin<"_Vector<4, double>(_Vector<4, double>, _Vector<4, double>, _Constant int)">;
  def vperm2f128_ps256 : X86Builtin<"_Vector<8, float>(_Vector<8, float>, _Vector<8, float>, _Constant int)">;
  def vperm2f128_si256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<8, int>, _Constant int)">;
}

let Features = "avx", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vpermilpd : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Constant int)">;
  def vpermilps : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Constant int)">;
}

let Features = "avx", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vpermilpd256 : X86Builtin<"_Vector<4, double>(_Vector<4, double>, _Constant int)">;
  def vpermilps256 : X86Builtin<"_Vector<8, float>(_Vector<8, float>, _Constant int)">;
  def vinsertf128_pd256 : X86Builtin<"_Vector<4, double>(_Vector<4, double>, _Vector<2, double>, _Constant int)">;
  def vinsertf128_ps256 : X86Builtin<"_Vector<8, float>(_Vector<8, float>, _Vector<4, float>, _Constant int)">;
  def vinsertf128_si256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<4, int>, _Constant int)">;
  def sqrtpd256 : X86Builtin<"_Vector<4, double>(_Vector<4, double>)">;
  def sqrtps256 : X86Builtin<"_Vector<8, float>(_Vector<8, float>)">;
  def rsqrtps256 : X86Builtin<"_Vector<8, float>(_Vector<8, float>)">;
  def rcpps256 : X86Builtin<"_Vector<8, float>(_Vector<8, float>)">;
  def roundpd256 : X86Builtin<"_Vector<4, double>(_Vector<4, double>, _Constant int)">;
  def roundps256 : X86Builtin<"_Vector<8, float>(_Vector<8, float>, _Constant int)">;
}

let Features = "avx", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vtestzpd : X86Builtin<"int(_Vector<2, double>, _Vector<2, double>)">;
  def vtestcpd : X86Builtin<"int(_Vector<2, double>, _Vector<2, double>)">;
  def vtestnzcpd : X86Builtin<"int(_Vector<2, double>, _Vector<2, double>)">;
  def vtestzps : X86Builtin<"int(_Vector<4, float>, _Vector<4, float>)">;
  def vtestcps : X86Builtin<"int(_Vector<4, float>, _Vector<4, float>)">;
  def vtestnzcps : X86Builtin<"int(_Vector<4, float>, _Vector<4, float>)">;
}

let Features = "avx", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vtestzpd256 : X86Builtin<"int(_Vector<4, double>, _Vector<4, double>)">;
  def vtestcpd256 : X86Builtin<"int(_Vector<4, double>, _Vector<4, double>)">;
  def vtestnzcpd256 : X86Builtin<"int(_Vector<4, double>, _Vector<4, double>)">;
  def vtestzps256 : X86Builtin<"int(_Vector<8, float>, _Vector<8, float>)">;
  def vtestcps256 : X86Builtin<"int(_Vector<8, float>, _Vector<8, float>)">;
  def vtestnzcps256 : X86Builtin<"int(_Vector<8, float>, _Vector<8, float>)">;
  def ptestz256 : X86Builtin<"int(_Vector<4, long long int>, _Vector<4, long long int>)">;
  def ptestc256 : X86Builtin<"int(_Vector<4, long long int>, _Vector<4, long long int>)">;
  def ptestnzc256 : X86Builtin<"int(_Vector<4, long long int>, _Vector<4, long long int>)">;
  def movmskpd256 : X86Builtin<"int(_Vector<4, double>)">;
  def movmskps256 : X86Builtin<"int(_Vector<8, float>)">;
}

let Features = "avx", Attributes = [NoThrow] in {
  def vzeroall : X86Builtin<"void()">;
  def vzeroupper : X86Builtin<"void()">;
}

let Features = "avx", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def lddqu256 : X86Builtin<"_Vector<32, char>(char const *)">;
}

let Features = "avx", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def maskloadpd : X86Builtin<"_Vector<2, double>(_Vector<2, double const *>, _Vector<2, long long int>)">;
  def maskloadps : X86Builtin<"_Vector<4, float>(_Vector<4, float const *>, _Vector<4, int>)">;
}

let Features = "avx", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def maskloadpd256 : X86Builtin<"_Vector<4, double>(_Vector<4, double const *>, _Vector<4, long long int>)">;
  def maskloadps256 : X86Builtin<"_Vector<8, float>(_Vector<8, float const *>, _Vector<8, int>)">;
}

let Features = "avx", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def maskstorepd : X86Builtin<"void(_Vector<2, double *>, _Vector<2, long long int>, _Vector<2, double>)">;
  def maskstoreps : X86Builtin<"void(_Vector<4, float *>, _Vector<4, int>, _Vector<4, float>)">;
}

let Features = "avx", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def maskstorepd256 : X86Builtin<"void(_Vector<4, double *>, _Vector<4, long long int>, _Vector<4, double>)">;
  def maskstoreps256 : X86Builtin<"void(_Vector<8, float *>, _Vector<8, int>, _Vector<8, float>)">;
}

let Features = "avx", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vec_ext_v32qi : X86Builtin<"char(_Vector<32, char>, _Constant int)">;
  def vec_ext_v16hi : X86Builtin<"short(_Vector<16, short>, _Constant int)">;
  def vec_ext_v8si : X86Builtin<"int(_Vector<8, int>, _Constant int)">;
  def vec_set_v32qi : X86Builtin<"_Vector<32, char>(_Vector<32, char>, char, _Constant int)">;
  def vec_set_v16hi : X86Builtin<"_Vector<16, short>(_Vector<16, short>, short, _Constant int)">;
  def vec_set_v8si : X86Builtin<"_Vector<8, int>(_Vector<8, int>, int, _Constant int)">;
}

let Features = "avx2", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def mpsadbw256 : X86Builtin<"_Vector<32, char>(_Vector<32, char>, _Vector<32, char>, _Constant char)">;
  def packsswb256 : X86Builtin<"_Vector<32, char>(_Vector<16, short>, _Vector<16, short>)">;
  def packssdw256 : X86Builtin<"_Vector<16, short>(_Vector<8, int>, _Vector<8, int>)">;
  def packuswb256 : X86Builtin<"_Vector<32, char>(_Vector<16, short>, _Vector<16, short>)">;
  def packusdw256 : X86Builtin<"_Vector<16, short>(_Vector<8, int>, _Vector<8, int>)">;
  def palignr256 : X86Builtin<"_Vector<32, char>(_Vector<32, char>, _Vector<32, char>, _Constant int)">;
  def pavgb256 : X86Builtin<"_Vector<32, char>(_Vector<32, char>, _Vector<32, char>)">;
  def pavgw256 : X86Builtin<"_Vector<16, short>(_Vector<16, short>, _Vector<16, short>)">;
  def pblendvb256 : X86Builtin<"_Vector<32, char>(_Vector<32, char>, _Vector<32, char>, _Vector<32, char>)">;
  def pblendw256 : X86Builtin<"_Vector<16, short>(_Vector<16, short>, _Vector<16, short>, _Constant int)">;
  def phaddw256 : X86Builtin<"_Vector<16, short>(_Vector<16, short>, _Vector<16, short>)">;
  def phaddd256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<8, int>)">;
  def phaddsw256 : X86Builtin<"_Vector<16, short>(_Vector<16, short>, _Vector<16, short>)">;
  def phsubw256 : X86Builtin<"_Vector<16, short>(_Vector<16, short>, _Vector<16, short>)">;
  def phsubd256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<8, int>)">;
  def phsubsw256 : X86Builtin<"_Vector<16, short>(_Vector<16, short>, _Vector<16, short>)">;
  def pmaddubsw256 : X86Builtin<"_Vector<16, short>(_Vector<32, char>, _Vector<32, char>)">;
  def pmaddwd256 : X86Builtin<"_Vector<8, int>(_Vector<16, short>, _Vector<16, short>)">;
  def pmovmskb256 : X86Builtin<"int(_Vector<32, char>)">;
  def pmuldq256 : X86Builtin<"_Vector<4, long long int>(_Vector<8, int>, _Vector<8, int>)">;
  def pmulhrsw256 : X86Builtin<"_Vector<16, short>(_Vector<16, short>, _Vector<16, short>)">;
  def pmuludq256 : X86Builtin<"_Vector<4, long long int>(_Vector<8, int>, _Vector<8, int>)">;
  def psadbw256 : X86Builtin<"_Vector<4, long long int>(_Vector<32, char>, _Vector<32, char>)">;
  def pshufb256 : X86Builtin<"_Vector<32, char>(_Vector<32, char>, _Vector<32, char>)">;
  def pshufd256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Constant int)">;
  def pshuflw256 : X86Builtin<"_Vector<16, short>(_Vector<16, short>, _Constant int)">;
  def pshufhw256 : X86Builtin<"_Vector<16, short>(_Vector<16, short>, _Constant int)">;
  def psignb256 : X86Builtin<"_Vector<32, char>(_Vector<32, char>, _Vector<32, char>)">;
  def psignw256 : X86Builtin<"_Vector<16, short>(_Vector<16, short>, _Vector<16, short>)">;
  def psignd256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<8, int>)">;
  def psllwi256 : X86Builtin<"_Vector<16, short>(_Vector<16, short>, int)">;
  def psllw256 : X86Builtin<"_Vector<16, short>(_Vector<16, short>, _Vector<8, short>)">;
  def pslldqi256_byteshift : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, _Constant int)">;
  def pslldi256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, int)">;
  def pslld256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<4, int>)">;
  def psllqi256 : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, int)">;
  def psllq256 : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, _Vector<2, long long int>)">;
  def psrawi256 : X86Builtin<"_Vector<16, short>(_Vector<16, short>, int)">;
  def psraw256 : X86Builtin<"_Vector<16, short>(_Vector<16, short>, _Vector<8, short>)">;
  def psradi256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, int)">;
  def psrad256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<4, int>)">;
  def psrldqi256_byteshift : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, _Constant int)">;
  def psrlwi256 : X86Builtin<"_Vector<16, short>(_Vector<16, short>, int)">;
  def psrlw256 : X86Builtin<"_Vector<16, short>(_Vector<16, short>, _Vector<8, short>)">;
  def psrldi256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, int)">;
  def psrld256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<4, int>)">;
  def psrlqi256 : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, int)">;
  def psrlq256 : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, _Vector<2, long long int>)">;
  def pblendd128 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>, _Constant int)">;
  def pblendd256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<8, int>, _Constant int)">;
  def permvarsi256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<8, int>)">;
  def permdf256 : X86Builtin<"_Vector<4, double>(_Vector<4, double>, _Constant int)">;
  def permvarsf256 : X86Builtin<"_Vector<8, float>(_Vector<8, float>, _Vector<8, int>)">;
  def permti256 : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, _Vector<4, long long int>, _Constant int)">;
  def permdi256 : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, _Constant int)">;
  def extract128i256 : X86Builtin<"_Vector<2, long long int>(_Vector<4, long long int>, _Constant int)">;
  def insert128i256 : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, _Vector<2, long long int>, _Constant int)">;
}

let Features = "avx2", Attributes = [NoThrow, Const, Constexpr, RequiredVectorWidth<256>] in {
  def pmulhuw256 : X86Builtin<"_Vector<16, unsigned short>(_Vector<16, unsigned short>, _Vector<16, unsigned short>)">;
  def pmulhw256 : X86Builtin<"_Vector<16, short>(_Vector<16, short>, _Vector<16, short>)">;
}

let Features = "avx2", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def maskloadd256 : X86Builtin<"_Vector<8, int>(_Vector<8, int const *>, _Vector<8, int>)">;
  def maskloadq256 : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int const *>, _Vector<4, long long int>)">;
}

let Features = "avx2", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def maskloadd : X86Builtin<"_Vector<4, int>(_Vector<4, int const *>, _Vector<4, int>)">;
  def maskloadq : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int const *>, _Vector<2, long long int>)">;
}

let Features = "avx2", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def maskstored256 : X86Builtin<"void(_Vector<8, int *>, _Vector<8, int>, _Vector<8, int>)">;
  def maskstoreq256 : X86Builtin<"void(_Vector<4, long long int *>, _Vector<4, long long int>, _Vector<4, long long int>)">;
}

let Features = "avx2", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def maskstored : X86Builtin<"void(_Vector<4, int *>, _Vector<4, int>, _Vector<4, int>)">;
  def maskstoreq : X86Builtin<"void(_Vector<2, long long int *>, _Vector<2, long long int>, _Vector<2, long long int>)">;
}

let Features = "avx2", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def psllv8si : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<8, int>)">;
}

let Features = "avx2", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def psllv4si : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>)">;
}

let Features = "avx2", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def psllv4di : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, _Vector<4, long long int>)">;
}

let Features = "avx2", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def psllv2di : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, _Vector<2, long long int>)">;
}

let Features = "avx2", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def psrav8si : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<8, int>)">;
}

let Features = "avx2", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def psrav4si : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>)">;
}

let Features = "avx2", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def psrlv8si : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<8, int>)">;
}

let Features = "avx2", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def psrlv4si : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>)">;
}

let Features = "avx2", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def psrlv4di : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, _Vector<4, long long int>)">;
}

let Features = "avx2", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def psrlv2di : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, _Vector<2, long long int>)">;
}

let Features = "avx2", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def gatherd_pd : X86Builtin<"_Vector<2, double>(_Vector<2, double>, double const *, _Vector<4, int>, _Vector<2, double>, _Constant char)">;
}

let Features = "avx2", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def gatherd_pd256 : X86Builtin<"_Vector<4, double>(_Vector<4, double>, double const *, _Vector<4, int>, _Vector<4, double>, _Constant char)">;
}

let Features = "avx2", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def gatherq_pd : X86Builtin<"_Vector<2, double>(_Vector<2, double>, double const *, _Vector<2, long long int>, _Vector<2, double>, _Constant char)">;
}

let Features = "avx2", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def gatherq_pd256 : X86Builtin<"_Vector<4, double>(_Vector<4, double>, double const *, _Vector<4, long long int>, _Vector<4, double>, _Constant char)">;
}

let Features = "avx2", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def gatherd_ps : X86Builtin<"_Vector<4, float>(_Vector<4, float>, float const *, _Vector<4, int>, _Vector<4, float>, _Constant char)">;
}

let Features = "avx2", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def gatherd_ps256 : X86Builtin<"_Vector<8, float>(_Vector<8, float>, float const *, _Vector<8, int>, _Vector<8, float>, _Constant char)">;
}

let Features = "avx2", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def gatherq_ps : X86Builtin<"_Vector<4, float>(_Vector<4, float>, float const *, _Vector<2, long long int>, _Vector<4, float>, _Constant char)">;
}

let Features = "avx2", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def gatherq_ps256 : X86Builtin<"_Vector<4, float>(_Vector<4, float>, float const *, _Vector<4, long long int>, _Vector<4, float>, _Constant char)">;
}

let Features = "avx2", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def gatherd_q : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, long long int const *, _Vector<4, int>, _Vector<2, long long int>, _Constant char)">;
}

let Features = "avx2", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def gatherd_q256 : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, long long int const *, _Vector<4, int>, _Vector<4, long long int>, _Constant char)">;
}

let Features = "avx2", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def gatherq_q : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, long long int const *, _Vector<2, long long int>, _Vector<2, long long int>, _Constant char)">;
}

let Features = "avx2", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def gatherq_q256 : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, long long int const *, _Vector<4, long long int>, _Vector<4, long long int>, _Constant char)">;
}

let Features = "avx2", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def gatherd_d : X86Builtin<"_Vector<4, int>(_Vector<4, int>, int const *, _Vector<4, int>, _Vector<4, int>, _Constant char)">;
}

let Features = "avx2", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def gatherd_d256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, int const *, _Vector<8, int>, _Vector<8, int>, _Constant char)">;
}

let Features = "avx2", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def gatherq_d : X86Builtin<"_Vector<4, int>(_Vector<4, int>, int const *, _Vector<2, long long int>, _Vector<4, int>, _Constant char)">;
}

let Features = "avx2", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def gatherq_d256 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, int const *, _Vector<4, long long int>, _Vector<4, int>, _Constant char)">;
}

let Features = "f16c", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vcvtps2ph : X86Builtin<"_Vector<8, short>(_Vector<4, float>, _Constant int)">;
}

let Features = "f16c", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vcvtps2ph256 : X86Builtin<"_Vector<8, short>(_Vector<8, float>, _Constant int)">;
}

let Features = "f16c", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vcvtph2ps : X86Builtin<"_Vector<4, float>(_Vector<8, short>)">;
}

let Features = "f16c", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vcvtph2ps256 : X86Builtin<"_Vector<8, float>(_Vector<8, short>)">;
}

let Features = "rdrnd", Attributes = [NoThrow] in {
  def rdrand16_step : X86Builtin<"unsigned int(unsigned short *)">;
  def rdrand32_step : X86Builtin<"unsigned int(unsigned int *)">;
}

let Features = "fxsr", Attributes = [NoThrow] in {
  def fxrstor : X86Builtin<"void(void *)">;
  def fxsave : X86Builtin<"void(void *)">;
}

let Features = "xsave", Attributes = [NoThrow] in {
  def xsave : X86Builtin<"void(void *, unsigned long long int)">;
  def xrstor : X86Builtin<"void(void *, unsigned long long int)">;
  def xgetbv : X86Builtin<"unsigned long long int(unsigned int)">;
}

let Header = "immintrin.h", Languages = "ALL_MS_LANGUAGES", Attributes = [NoThrow, RequireDeclaration] in {
  def _xgetbv : X86LibBuiltin<"uint64_t(unsigned int)">;
}

let Features = "xsave", Attributes = [NoThrow] in {
  def xsetbv : X86Builtin<"void(unsigned int, unsigned long long int)">;
}

let Header = "immintrin.h", Languages = "ALL_MS_LANGUAGES", Attributes = [NoThrow, RequireDeclaration] in {
  def _xsetbv : X86LibBuiltin<"void(unsigned int, uint64_t)">;
}

let Features = "xsaveopt", Attributes = [NoThrow] in {
  def xsaveopt : X86Builtin<"void(void *, unsigned long long int)">;
}

let Features = "xsaves", Attributes = [NoThrow] in {
  def xrstors : X86Builtin<"void(void *, unsigned long long int)">;
}

let Features = "xsavec", Attributes = [NoThrow] in {
  def xsavec : X86Builtin<"void(void *, unsigned long long int)">;
}

let Features = "xsaves", Attributes = [NoThrow] in {
  def xsaves : X86Builtin<"void(void *, unsigned long long int)">;
}

let Features = "shstk", Attributes = [NoThrow] in {
  def incsspd : X86Builtin<"void(unsigned int)">;
  def rdsspd : X86Builtin<"unsigned int(unsigned int)">;
  def saveprevssp : X86Builtin<"void()">;
  def rstorssp : X86Builtin<"void(void *)">;
  def wrssd : X86Builtin<"void(unsigned int, void *)">;
  def wrussd : X86Builtin<"void(unsigned int, void *)">;
  def setssbsy : X86Builtin<"void()">;
  def clrssbsy : X86Builtin<"void(void *)">;
}

let Features = "clflushopt", Attributes = [NoThrow] in {
  def clflushopt : X86Builtin<"void(void const *)">;
}

let Features = "clwb", Attributes = [NoThrow] in {
  def clwb : X86Builtin<"void(void const *)">;
}

let Attributes = [NoThrow] in {
  def wbinvd : X86Builtin<"void()">;
}

let Features = "wbnoinvd", Attributes = [NoThrow] in {
  def wbnoinvd : X86Builtin<"void()">;
}

let Attributes = [NoThrow, Constexpr] in {
  def addcarryx_u32 : X86Builtin<"unsigned char(unsigned char, unsigned int, unsigned int, unsigned int *)">;
  def subborrow_u32 : X86Builtin<"unsigned char(unsigned char, unsigned int, unsigned int, unsigned int *)">;
}

let Features = "rdseed", Attributes = [NoThrow] in {
  def rdseed16_step : X86Builtin<"unsigned int(unsigned short *)">;
  def rdseed32_step : X86Builtin<"unsigned int(unsigned int *)">;
}

let Features = "bmi", Attributes = [NoThrow, Const, Constexpr] in {
  def bextr_u32 : X86Builtin<"unsigned int(unsigned int, unsigned int)">;
}

let Attributes = [NoThrow, Const, Constexpr] in {
  def tzcnt_u16 : X86Builtin<"unsigned short(unsigned short)">;
  def tzcnt_u32 : X86Builtin<"unsigned int(unsigned int)">;
  def lzcnt_u16 : X86Builtin<"unsigned short(unsigned short)">;
  def lzcnt_u32 : X86Builtin<"unsigned int(unsigned int)">;
}

let Features = "bmi2", Attributes = [NoThrow, Const, Constexpr] in {
  def bzhi_si : X86Builtin<"unsigned int(unsigned int, unsigned int)">;
  def pdep_si : X86Builtin<"unsigned int(unsigned int, unsigned int)">;
  def pext_si : X86Builtin<"unsigned int(unsigned int, unsigned int)">;
}

let Features = "tbm", Attributes = [NoThrow, Const, Constexpr] in {
  def bextri_u32 : X86Builtin<"unsigned int(unsigned int, _Constant unsigned int)">;
}

let Features = "lwp", Attributes = [NoThrow] in {
  def llwpcb : X86Builtin<"void(void *)">;
  def slwpcb : X86Builtin<"void *()">;
  def lwpins32 : X86Builtin<"unsigned char(unsigned int, unsigned int, _Constant unsigned int)">;
  def lwpval32 : X86Builtin<"void(unsigned int, unsigned int, _Constant unsigned int)">;
}

let Features = "sha", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def sha1rnds4 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>, _Constant char)">;
  def sha1nexte : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>)">;
  def sha1msg1 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>)">;
  def sha1msg2 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>)">;
  def sha256rnds2 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>, _Vector<4, int>)">;
  def sha256msg1 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>)">;
  def sha256msg2 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>)">;
}

let Features = "fma", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vfmaddss3 : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>)">;
  def vfmaddsd3 : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Vector<2, double>)">;
}

let Features = "fma4", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vfmaddss : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>)">;
  def vfmaddsd : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Vector<2, double>)">;
}

let Features = "fma|fma4", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vfmaddsubps : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>)">;
  def vfmaddsubpd : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Vector<2, double>)">;
}

let Features = "fma|fma4",
    Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vfmaddsubps256 : X86Builtin<"_Vector<8, float>(_Vector<8, float>, _Vector<8, float>, _Vector<8, float>)">;
  def vfmaddsubpd256 : X86Builtin<"_Vector<4, double>(_Vector<4, double>, _Vector<4, double>, _Vector<4, double>)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vfmaddpd512_mask : X86Builtin<"_Vector<8, double>(_Vector<8, double>, _Vector<8, double>, _Vector<8, double>, unsigned char, _Constant int)">;
  def vfmaddpd512_maskz : X86Builtin<"_Vector<8, double>(_Vector<8, double>, _Vector<8, double>, _Vector<8, double>, unsigned char, _Constant int)">;
  def vfmaddpd512_mask3 : X86Builtin<"_Vector<8, double>(_Vector<8, double>, _Vector<8, double>, _Vector<8, double>, unsigned char, _Constant int)">;
  def vfmsubpd512_mask3 : X86Builtin<"_Vector<8, double>(_Vector<8, double>, _Vector<8, double>, _Vector<8, double>, unsigned char, _Constant int)">;
  def vfmaddps512_mask : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Vector<16, float>, _Vector<16, float>, unsigned short, _Constant int)">;
  def vfmaddps512_maskz : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Vector<16, float>, _Vector<16, float>, unsigned short, _Constant int)">;
  def vfmaddps512_mask3 : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Vector<16, float>, _Vector<16, float>, unsigned short, _Constant int)">;
  def vfmsubps512_mask3 : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Vector<16, float>, _Vector<16, float>, unsigned short, _Constant int)">;
  def vfmaddsubpd512_mask : X86Builtin<"_Vector<8, double>(_Vector<8, double>, _Vector<8, double>, _Vector<8, double>, unsigned char, _Constant int)">;
  def vfmaddsubpd512_maskz : X86Builtin<"_Vector<8, double>(_Vector<8, double>, _Vector<8, double>, _Vector<8, double>, unsigned char, _Constant int)">;
  def vfmaddsubpd512_mask3 : X86Builtin<"_Vector<8, double>(_Vector<8, double>, _Vector<8, double>, _Vector<8, double>, unsigned char, _Constant int)">;
  def vfmsubaddpd512_mask3 : X86Builtin<"_Vector<8, double>(_Vector<8, double>, _Vector<8, double>, _Vector<8, double>, unsigned char, _Constant int)">;
  def vfmaddsubps512_mask : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Vector<16, float>, _Vector<16, float>, unsigned short, _Constant int)">;
  def vfmaddsubps512_maskz : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Vector<16, float>, _Vector<16, float>, unsigned short, _Constant int)">;
  def vfmaddsubps512_mask3 : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Vector<16, float>, _Vector<16, float>, unsigned short, _Constant int)">;
  def vfmsubaddps512_mask3 : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Vector<16, float>, _Vector<16, float>, unsigned short, _Constant int)">;
}

let Features = "xop", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vpmacssww : X86Builtin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>, _Vector<8, short>)">;
  def vpmacsww : X86Builtin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>, _Vector<8, short>)">;
  def vpmacsswd : X86Builtin<"_Vector<4, int>(_Vector<8, short>, _Vector<8, short>, _Vector<4, int>)">;
  def vpmacswd : X86Builtin<"_Vector<4, int>(_Vector<8, short>, _Vector<8, short>, _Vector<4, int>)">;
  def vpmacssdd : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>, _Vector<4, int>)">;
  def vpmacsdd : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>, _Vector<4, int>)">;
  def vpmacssdql : X86Builtin<"_Vector<2, long long int>(_Vector<4, int>, _Vector<4, int>, _Vector<2, long long int>)">;
  def vpmacsdql : X86Builtin<"_Vector<2, long long int>(_Vector<4, int>, _Vector<4, int>, _Vector<2, long long int>)">;
  def vpmacssdqh : X86Builtin<"_Vector<2, long long int>(_Vector<4, int>, _Vector<4, int>, _Vector<2, long long int>)">;
  def vpmacsdqh : X86Builtin<"_Vector<2, long long int>(_Vector<4, int>, _Vector<4, int>, _Vector<2, long long int>)">;
  def vpmadcsswd : X86Builtin<"_Vector<4, int>(_Vector<8, short>, _Vector<8, short>, _Vector<4, int>)">;
  def vpmadcswd : X86Builtin<"_Vector<4, int>(_Vector<8, short>, _Vector<8, short>, _Vector<4, int>)">;
  def vphaddbw : X86Builtin<"_Vector<8, short>(_Vector<16, char>)">;
  def vphaddbd : X86Builtin<"_Vector<4, int>(_Vector<16, char>)">;
  def vphaddbq : X86Builtin<"_Vector<2, long long int>(_Vector<16, char>)">;
  def vphaddwd : X86Builtin<"_Vector<4, int>(_Vector<8, short>)">;
  def vphaddwq : X86Builtin<"_Vector<2, long long int>(_Vector<8, short>)">;
  def vphadddq : X86Builtin<"_Vector<2, long long int>(_Vector<4, int>)">;
  def vphaddubw : X86Builtin<"_Vector<8, short>(_Vector<16, char>)">;
  def vphaddubd : X86Builtin<"_Vector<4, int>(_Vector<16, char>)">;
  def vphaddubq : X86Builtin<"_Vector<2, long long int>(_Vector<16, char>)">;
  def vphadduwd : X86Builtin<"_Vector<4, int>(_Vector<8, short>)">;
  def vphadduwq : X86Builtin<"_Vector<2, long long int>(_Vector<8, short>)">;
  def vphaddudq : X86Builtin<"_Vector<2, long long int>(_Vector<4, int>)">;
  def vphsubbw : X86Builtin<"_Vector<8, short>(_Vector<16, char>)">;
  def vphsubwd : X86Builtin<"_Vector<4, int>(_Vector<8, short>)">;
  def vphsubdq : X86Builtin<"_Vector<2, long long int>(_Vector<4, int>)">;
  def vpperm : X86Builtin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>)">;
  def vprotb : X86Builtin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>)">;
  def vprotw : X86Builtin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>)">;
  def vprotd : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>)">;
  def vprotq : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, _Vector<2, long long int>)">;
  def vprotbi : X86Builtin<"_Vector<16, char>(_Vector<16, char>, _Constant char)">;
  def vprotwi : X86Builtin<"_Vector<8, short>(_Vector<8, short>, _Constant char)">;
  def vprotdi : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Constant char)">;
  def vprotqi : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, _Constant char)">;
  def vpshlb : X86Builtin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>)">;
  def vpshlw : X86Builtin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>)">;
  def vpshld : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>)">;
  def vpshlq : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, _Vector<2, long long int>)">;
  def vpshab : X86Builtin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>)">;
  def vpshaw : X86Builtin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>)">;
  def vpshad : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>)">;
  def vpshaq : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, _Vector<2, long long int>)">;
  def vpcomub : X86Builtin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>, _Constant char)">;
  def vpcomuw : X86Builtin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>, _Constant char)">;
  def vpcomud : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>, _Constant char)">;
  def vpcomuq : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, _Vector<2, long long int>, _Constant char)">;
  def vpcomb : X86Builtin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>, _Constant char)">;
  def vpcomw : X86Builtin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>, _Constant char)">;
  def vpcomd : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>, _Constant char)">;
  def vpcomq : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, _Vector<2, long long int>, _Constant char)">;
  def vpermil2pd : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Vector<2, long long int>, _Constant char)">;
}

let Features = "xop", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vpermil2pd256 : X86Builtin<"_Vector<4, double>(_Vector<4, double>, _Vector<4, double>, _Vector<4, long long int>, _Constant char)">;
}

let Features = "xop", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vpermil2ps : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, int>, _Constant char)">;
}

let Features = "xop", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vpermil2ps256 : X86Builtin<"_Vector<8, float>(_Vector<8, float>, _Vector<8, float>, _Vector<8, int>, _Constant char)">;
}

let Features = "xop", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vfrczss : X86Builtin<"_Vector<4, float>(_Vector<4, float>)">;
  def vfrczsd : X86Builtin<"_Vector<2, double>(_Vector<2, double>)">;
  def vfrczps : X86Builtin<"_Vector<4, float>(_Vector<4, float>)">;
  def vfrczpd : X86Builtin<"_Vector<2, double>(_Vector<2, double>)">;
}

let Features = "xop", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vfrczps256 : X86Builtin<"_Vector<8, float>(_Vector<8, float>)">;
  def vfrczpd256 : X86Builtin<"_Vector<4, double>(_Vector<4, double>)">;
}

let Features = "rtm", Attributes = [NoThrow] in {
  def xbegin : X86Builtin<"int()">;
  def xend : X86Builtin<"void()">;
  def xabort : X86Builtin<"void(_Constant char)">;
  def xtest : X86Builtin<"int()">;
}

let Features = "rdpid", Attributes = [NoThrow] in {
  def rdpid : X86Builtin<"unsigned int()">;
}

let Features = "rdpru", Attributes = [NoThrow], EnableOpenCLLong = 0 in {
  def rdpru : X86Builtin<"unsigned long long int(int)">;
}

let Features = "pku", Attributes = [NoThrow] in {
  def rdpkru : X86Builtin<"unsigned int()">;
  def wrpkru : X86Builtin<"void(unsigned int)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def sqrtpd512 : X86Builtin<"_Vector<8, double>(_Vector<8, double>, _Constant int)">;
  def sqrtps512 : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Constant int)">;
}

let Features = "avx512f", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def rsqrt14sd_mask : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Vector<2, double>, unsigned char)">;
  def rsqrt14ss_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>, unsigned char)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def rsqrt14pd512_mask : X86Builtin<"_Vector<8, double>(_Vector<8, double>, _Vector<8, double>, unsigned char)">;
  def rsqrt14ps512_mask : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Vector<16, float>, unsigned short)">;
}

let Features = "avx512f", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def rcp14sd_mask : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Vector<2, double>, unsigned char)">;
  def rcp14ss_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>, unsigned char)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def rcp14pd512_mask : X86Builtin<"_Vector<8, double>(_Vector<8, double>, _Vector<8, double>, unsigned char)">;
  def rcp14ps512_mask : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Vector<16, float>, unsigned short)">;
  def cvttps2dq512_mask : X86Builtin<"_Vector<16, int>(_Vector<16, float>, _Vector<16, int>, unsigned short, _Constant int)">;
  def cvttps2udq512_mask : X86Builtin<"_Vector<16, int>(_Vector<16, float>, _Vector<16, int>, unsigned short, _Constant int)">;
  def cvttpd2dq512_mask : X86Builtin<"_Vector<8, int>(_Vector<8, double>, _Vector<8, int>, unsigned char, _Constant int)">;
  def cvttpd2udq512_mask : X86Builtin<"_Vector<8, int>(_Vector<8, double>, _Vector<8, int>, unsigned char, _Constant int)">;
  def cmpps512_mask : X86Builtin<"unsigned short(_Vector<16, float>, _Vector<16, float>, _Constant int, unsigned short, _Constant int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def cmpps256_mask : X86Builtin<"unsigned char(_Vector<8, float>, _Vector<8, float>, _Constant int, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def cmpps128_mask : X86Builtin<"unsigned char(_Vector<4, float>, _Vector<4, float>, _Constant int, unsigned char)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def cmppd512_mask : X86Builtin<"unsigned char(_Vector<8, double>, _Vector<8, double>, _Constant int, unsigned char, _Constant int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def cmppd256_mask : X86Builtin<"unsigned char(_Vector<4, double>, _Vector<4, double>, _Constant int, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def cmppd128_mask : X86Builtin<"unsigned char(_Vector<2, double>, _Vector<2, double>, _Constant int, unsigned char)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def rndscaleps_mask : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Constant int, _Vector<16, float>, unsigned short, _Constant int)">;
  def rndscalepd_mask : X86Builtin<"_Vector<8, double>(_Vector<8, double>, _Constant int, _Vector<8, double>, unsigned char, _Constant int)">;
  def cvtps2dq512_mask : X86Builtin<"_Vector<16, int>(_Vector<16, float>, _Vector<16, int>, unsigned short, _Constant int)">;
  def cvtpd2dq512_mask : X86Builtin<"_Vector<8, int>(_Vector<8, double>, _Vector<8, int>, unsigned char, _Constant int)">;
  def cvtps2udq512_mask : X86Builtin<"_Vector<16, int>(_Vector<16, float>, _Vector<16, int>, unsigned short, _Constant int)">;
  def cvtpd2udq512_mask : X86Builtin<"_Vector<8, int>(_Vector<8, double>, _Vector<8, int>, unsigned char, _Constant int)">;
  def minps512 : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Vector<16, float>, _Constant int)">;
  def minpd512 : X86Builtin<"_Vector<8, double>(_Vector<8, double>, _Vector<8, double>, _Constant int)">;
  def maxps512 : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Vector<16, float>, _Constant int)">;
  def maxpd512 : X86Builtin<"_Vector<8, double>(_Vector<8, double>, _Vector<8, double>, _Constant int)">;
  def cvtdq2ps512_mask : X86Builtin<"_Vector<16, float>(_Vector<16, int>, _Vector<16, float>, unsigned short, _Constant int)">;
  def cvtudq2ps512_mask : X86Builtin<"_Vector<16, float>(_Vector<16, int>, _Vector<16, float>, unsigned short, _Constant int)">;
  def cvtpd2ps512_mask : X86Builtin<"_Vector<8, float>(_Vector<8, double>, _Vector<8, float>, unsigned char, _Constant int)">;
  def vcvtps2ph512_mask : X86Builtin<"_Vector<16, short>(_Vector<16, float>, _Constant int, _Vector<16, short>, unsigned short)">;
  def vcvtph2ps512_mask : X86Builtin<"_Vector<16, float>(_Vector<16, short>, _Vector<16, float>, unsigned short, _Constant int)">;
  def pmuldq512 : X86Builtin<"_Vector<8, long long int>(_Vector<16, int>, _Vector<16, int>)">;
  def pmuludq512 : X86Builtin<"_Vector<8, long long int>(_Vector<16, int>, _Vector<16, int>)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def loaddqusi512_mask : X86Builtin<"_Vector<16, int>(int const *, _Vector<16, int>, unsigned short)">;
  def loaddqudi512_mask : X86Builtin<"_Vector<8, long long int>(long long int const *, _Vector<8, long long int>, unsigned char)">;
  def loadups512_mask : X86Builtin<"_Vector<16, float>(float const *, _Vector<16, float>, unsigned short)">;
  def loadaps512_mask : X86Builtin<"_Vector<16, float>(_Vector<16, float const *>, _Vector<16, float>, unsigned short)">;
  def loadupd512_mask : X86Builtin<"_Vector<8, double>(double const *, _Vector<8, double>, unsigned char)">;
  def loadapd512_mask : X86Builtin<"_Vector<8, double>(_Vector<8, double const *>, _Vector<8, double>, unsigned char)">;
  def storedqudi512_mask : X86Builtin<"void(long long int *, _Vector<8, long long int>, unsigned char)">;
  def storedqusi512_mask : X86Builtin<"void(int *, _Vector<16, int>, unsigned short)">;
  def storeupd512_mask : X86Builtin<"void(double *, _Vector<8, double>, unsigned char)">;
  def storeapd512_mask : X86Builtin<"void(_Vector<8, double *>, _Vector<8, double>, unsigned char)">;
  def storeups512_mask : X86Builtin<"void(float *, _Vector<16, float>, unsigned short)">;
  def storeaps512_mask : X86Builtin<"void(_Vector<16, float *>, _Vector<16, float>, unsigned short)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def alignq512 : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int>, _Vector<8, long long int>, _Constant int)">;
  def alignd512 : X86Builtin<"_Vector<16, int>(_Vector<16, int>, _Vector<16, int>, _Constant int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def alignd128 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>, _Constant int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def alignd256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<8, int>, _Constant int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def alignq128 : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, _Vector<2, long long int>, _Constant int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def alignq256 : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, _Vector<4, long long int>, _Constant int)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def extractf64x4_mask : X86Builtin<"_Vector<4, double>(_Vector<8, double>, _Constant int, _Vector<4, double>, unsigned char)">;
  def extractf32x4_mask : X86Builtin<"_Vector<4, float>(_Vector<16, float>, _Constant int, _Vector<4, float>, unsigned char)">;
}

let Features = "avx512vl,avx512vnni|avxvnni", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vpdpbusd128 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>, _Vector<4, int>)">;
}

let Features = "avx512vl,avx512vnni|avxvnni", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vpdpbusd256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<8, int>, _Vector<8, int>)">;
}

let Features = "avx512vnni,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vpdpbusd512 : X86Builtin<"_Vector<16, int>(_Vector<16, int>, _Vector<16, int>, _Vector<16, int>)">;
}

let Features = "avx512vl,avx512vnni|avxvnni", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vpdpbusds128 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>, _Vector<4, int>)">;
}

let Features = "avx512vl,avx512vnni|avxvnni", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vpdpbusds256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<8, int>, _Vector<8, int>)">;
}

let Features = "avx512vnni,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vpdpbusds512 : X86Builtin<"_Vector<16, int>(_Vector<16, int>, _Vector<16, int>, _Vector<16, int>)">;
}

let Features = "avx512vl,avx512vnni|avxvnni", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vpdpwssd128 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>, _Vector<4, int>)">;
}

let Features = "avx512vl,avx512vnni|avxvnni", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vpdpwssd256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<8, int>, _Vector<8, int>)">;
}

let Features = "avx512vnni,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vpdpwssd512 : X86Builtin<"_Vector<16, int>(_Vector<16, int>, _Vector<16, int>, _Vector<16, int>)">;
}

let Features = "avx512vl,avx512vnni|avxvnni", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vpdpwssds128 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>, _Vector<4, int>)">;
}

let Features = "avx512vl,avx512vnni|avxvnni", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vpdpwssds256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<8, int>, _Vector<8, int>)">;
}

let Features = "avx512vnni,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vpdpwssds512 : X86Builtin<"_Vector<16, int>(_Vector<16, int>, _Vector<16, int>, _Vector<16, int>)">;
}

let Features = "avxvnniint8|avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vpdpbssd128 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>, _Vector<4, int>)">;
}

let Features = "avxvnniint8|avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vpdpbssd256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<8, int>, _Vector<8, int>)">;
}

let Features = "avxvnniint8|avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vpdpbssds128 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>, _Vector<4, int>)">;
}

let Features = "avxvnniint8|avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vpdpbssds256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<8, int>, _Vector<8, int>)">;
}

let Features = "avxvnniint8|avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vpdpbsud128 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>, _Vector<4, int>)">;
}

let Features = "avxvnniint8|avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vpdpbsud256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<8, int>, _Vector<8, int>)">;
}

let Features = "avxvnniint8|avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vpdpbsuds128 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>, _Vector<4, int>)">;
}

let Features = "avxvnniint8|avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vpdpbsuds256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<8, int>, _Vector<8, int>)">;
}

let Features = "avxvnniint8|avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vpdpbuud128 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>, _Vector<4, int>)">;
}

let Features = "avxvnniint8|avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vpdpbuud256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<8, int>, _Vector<8, int>)">;
}

let Features = "avxvnniint8|avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vpdpbuuds128 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>, _Vector<4, int>)">;
}

let Features = "avxvnniint8|avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vpdpbuuds256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<8, int>, _Vector<8, int>)">;
}

let Features = "movrs", Attributes = [NoThrow, Const] in {
  def prefetchrs : X86Builtin<"void(void const *)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def gather3div2df : X86Builtin<"_Vector<2, double>(_Vector<2, double>, void const *, _Vector<2, long long int>, unsigned char, _Constant int)">;
  def gather3div2di : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, void const *, _Vector<2, long long int>, unsigned char, _Constant int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def gather3div4df : X86Builtin<"_Vector<4, double>(_Vector<4, double>, void const *, _Vector<4, long long int>, unsigned char, _Constant int)">;
  def gather3div4di : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, void const *, _Vector<4, long long int>, unsigned char, _Constant int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def gather3div4sf : X86Builtin<"_Vector<4, float>(_Vector<4, float>, void const *, _Vector<2, long long int>, unsigned char, _Constant int)">;
  def gather3div4si : X86Builtin<"_Vector<4, int>(_Vector<4, int>, void const *, _Vector<2, long long int>, unsigned char, _Constant int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def gather3div8sf : X86Builtin<"_Vector<4, float>(_Vector<4, float>, void const *, _Vector<4, long long int>, unsigned char, _Constant int)">;
  def gather3div8si : X86Builtin<"_Vector<4, int>(_Vector<4, int>, void const *, _Vector<4, long long int>, unsigned char, _Constant int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def gather3siv2df : X86Builtin<"_Vector<2, double>(_Vector<2, double>, void const *, _Vector<4, int>, unsigned char, _Constant int)">;
  def gather3siv2di : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, void const *, _Vector<4, int>, unsigned char, _Constant int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def gather3siv4df : X86Builtin<"_Vector<4, double>(_Vector<4, double>, void const *, _Vector<4, int>, unsigned char, _Constant int)">;
  def gather3siv4di : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, void const *, _Vector<4, int>, unsigned char, _Constant int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def gather3siv4sf : X86Builtin<"_Vector<4, float>(_Vector<4, float>, void const *, _Vector<4, int>, unsigned char, _Constant int)">;
  def gather3siv4si : X86Builtin<"_Vector<4, int>(_Vector<4, int>, void const *, _Vector<4, int>, unsigned char, _Constant int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def gather3siv8sf : X86Builtin<"_Vector<8, float>(_Vector<8, float>, void const *, _Vector<8, int>, unsigned char, _Constant int)">;
  def gather3siv8si : X86Builtin<"_Vector<8, int>(_Vector<8, int>, void const *, _Vector<8, int>, unsigned char, _Constant int)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def gathersiv8df : X86Builtin<"_Vector<8, double>(_Vector<8, double>, void const *, _Vector<8, int>, unsigned char, _Constant int)">;
  def gathersiv16sf : X86Builtin<"_Vector<16, float>(_Vector<16, float>, void const *, _Vector<16, int>, unsigned short, _Constant int)">;
  def gatherdiv8df : X86Builtin<"_Vector<8, double>(_Vector<8, double>, void const *, _Vector<8, long long int>, unsigned char, _Constant int)">;
  def gatherdiv16sf : X86Builtin<"_Vector<8, float>(_Vector<8, float>, void const *, _Vector<8, long long int>, unsigned char, _Constant int)">;
  def gathersiv8di : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int>, void const *, _Vector<8, int>, unsigned char, _Constant int)">;
  def gathersiv16si : X86Builtin<"_Vector<16, int>(_Vector<16, int>, void const *, _Vector<16, int>, unsigned short, _Constant int)">;
  def gatherdiv8di : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int>, void const *, _Vector<8, long long int>, unsigned char, _Constant int)">;
  def gatherdiv16si : X86Builtin<"_Vector<8, int>(_Vector<8, int>, void const *, _Vector<8, long long int>, unsigned char, _Constant int)">;
  def scattersiv8df : X86Builtin<"void(void *, unsigned char, _Vector<8, int>, _Vector<8, double>, _Constant int)">;
  def scattersiv16sf : X86Builtin<"void(void *, unsigned short, _Vector<16, int>, _Vector<16, float>, _Constant int)">;
  def scatterdiv8df : X86Builtin<"void(void *, unsigned char, _Vector<8, long long int>, _Vector<8, double>, _Constant int)">;
  def scatterdiv16sf : X86Builtin<"void(void *, unsigned char, _Vector<8, long long int>, _Vector<8, float>, _Constant int)">;
  def scattersiv8di : X86Builtin<"void(void *, unsigned char, _Vector<8, int>, _Vector<8, long long int>, _Constant int)">;
  def scattersiv16si : X86Builtin<"void(void *, unsigned short, _Vector<16, int>, _Vector<16, int>, _Constant int)">;
  def scatterdiv8di : X86Builtin<"void(void *, unsigned char, _Vector<8, long long int>, _Vector<8, long long int>, _Constant int)">;
  def scatterdiv16si : X86Builtin<"void(void *, unsigned char, _Vector<8, long long int>, _Vector<8, int>, _Constant int)">;
}

let Features = "avx512dq", Attributes = [NoThrow, Const] in {
  def knotqi : X86Builtin<"unsigned char(unsigned char)">;
}

let Features = "avx512f", Attributes = [NoThrow, Const] in {
  def knothi : X86Builtin<"unsigned short(unsigned short)">;
}

let Features = "avx512bw", Attributes = [NoThrow, Const] in {
  def knotsi : X86Builtin<"unsigned int(unsigned int)">;
  def knotdi : X86Builtin<"unsigned long long int(unsigned long long int)">;
}

let Features = "avx512vl,avx512bw", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def cmpb128_mask : X86Builtin<"unsigned short(_Vector<16, char>, _Vector<16, char>, _Constant int, unsigned short)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def cmpd128_mask : X86Builtin<"unsigned char(_Vector<4, int>, _Vector<4, int>, _Constant int, unsigned char)">;
  def cmpq128_mask : X86Builtin<"unsigned char(_Vector<2, long long int>, _Vector<2, long long int>, _Constant int, unsigned char)">;
}

let Features = "avx512vl,avx512bw", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def cmpw128_mask : X86Builtin<"unsigned char(_Vector<8, short>, _Vector<8, short>, _Constant int, unsigned char)">;
}

let Features = "avx512vl,avx512bw", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def cmpb256_mask : X86Builtin<"unsigned int(_Vector<32, char>, _Vector<32, char>, _Constant int, unsigned int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def cmpd256_mask : X86Builtin<"unsigned char(_Vector<8, int>, _Vector<8, int>, _Constant int, unsigned char)">;
  def cmpq256_mask : X86Builtin<"unsigned char(_Vector<4, long long int>, _Vector<4, long long int>, _Constant int, unsigned char)">;
}

let Features = "avx512vl,avx512bw", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def cmpw256_mask : X86Builtin<"unsigned short(_Vector<16, short>, _Vector<16, short>, _Constant int, unsigned short)">;
}

let Features = "avx512bw,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def cmpb512_mask : X86Builtin<"unsigned long long int(_Vector<64, char>, _Vector<64, char>, _Constant int, unsigned long long int)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def cmpd512_mask : X86Builtin<"unsigned short(_Vector<16, int>, _Vector<16, int>, _Constant int, unsigned short)">;
  def cmpq512_mask : X86Builtin<"unsigned char(_Vector<8, long long int>, _Vector<8, long long int>, _Constant int, unsigned char)">;
}

let Features = "avx512bw,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def cmpw512_mask : X86Builtin<"unsigned int(_Vector<32, short>, _Vector<32, short>, _Constant int, unsigned int)">;
}

let Features = "avx512vl,avx512bw", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def ucmpb128_mask : X86Builtin<"unsigned short(_Vector<16, char>, _Vector<16, char>, _Constant int, unsigned short)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def ucmpd128_mask : X86Builtin<"unsigned char(_Vector<4, int>, _Vector<4, int>, _Constant int, unsigned char)">;
  def ucmpq128_mask : X86Builtin<"unsigned char(_Vector<2, long long int>, _Vector<2, long long int>, _Constant int, unsigned char)">;
}

let Features = "avx512vl,avx512bw", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def ucmpw128_mask : X86Builtin<"unsigned char(_Vector<8, short>, _Vector<8, short>, _Constant int, unsigned char)">;
}

let Features = "avx512vl,avx512bw", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def ucmpb256_mask : X86Builtin<"unsigned int(_Vector<32, char>, _Vector<32, char>, _Constant int, unsigned int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def ucmpd256_mask : X86Builtin<"unsigned char(_Vector<8, int>, _Vector<8, int>, _Constant int, unsigned char)">;
  def ucmpq256_mask : X86Builtin<"unsigned char(_Vector<4, long long int>, _Vector<4, long long int>, _Constant int, unsigned char)">;
}

let Features = "avx512vl,avx512bw", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def ucmpw256_mask : X86Builtin<"unsigned short(_Vector<16, short>, _Vector<16, short>, _Constant int, unsigned short)">;
}

let Features = "avx512bw,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def ucmpb512_mask : X86Builtin<"unsigned long long int(_Vector<64, char>, _Vector<64, char>, _Constant int, unsigned long long int)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def ucmpd512_mask : X86Builtin<"unsigned short(_Vector<16, int>, _Vector<16, int>, _Constant int, unsigned short)">;
  def ucmpq512_mask : X86Builtin<"unsigned char(_Vector<8, long long int>, _Vector<8, long long int>, _Constant int, unsigned char)">;
}

let Features = "avx512bw,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def ucmpw512_mask : X86Builtin<"unsigned int(_Vector<32, short>, _Vector<32, short>, _Constant int, unsigned int)">;
  def packssdw512 : X86Builtin<"_Vector<32, short>(_Vector<16, int>, _Vector<16, int>)">;
  def packsswb512 : X86Builtin<"_Vector<64, char>(_Vector<32, short>, _Vector<32, short>)">;
  def packusdw512 : X86Builtin<"_Vector<32, short>(_Vector<16, int>, _Vector<16, int>)">;
  def packuswb512 : X86Builtin<"_Vector<64, char>(_Vector<32, short>, _Vector<32, short>)">;
  def pavgb512 : X86Builtin<"_Vector<64, char>(_Vector<64, char>, _Vector<64, char>)">;
  def pavgw512 : X86Builtin<"_Vector<32, short>(_Vector<32, short>, _Vector<32, short>)">;
  def pshufb512 : X86Builtin<"_Vector<64, char>(_Vector<64, char>, _Vector<64, char>)">;
}

let Features = "avx512cd,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vpconflictdi_128 : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>)">;
}

let Features = "avx512cd,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vpconflictdi_256 : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>)">;
}

let Features = "avx512cd,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vpconflictsi_128 : X86Builtin<"_Vector<4, int>(_Vector<4, int>)">;
}

let Features = "avx512cd,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vpconflictsi_256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>)">;
}

let Features = "avx512cd,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vpconflictdi_512 : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int>)">;
  def vpconflictsi_512 : X86Builtin<"_Vector<16, int>(_Vector<16, int>)">;
  def vplzcntd_512 : X86Builtin<"_Vector<16, int>(_Vector<16, int>)">;
  def vplzcntq_512 : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int>)">;
}

let Features = "avx512vl,avx512bitalg", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vpshufbitqmb128_mask : X86Builtin<"unsigned short(_Vector<16, char>, _Vector<16, char>, unsigned short)">;
}

let Features = "avx512vl,avx512bitalg", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vpshufbitqmb256_mask : X86Builtin<"unsigned int(_Vector<32, char>, _Vector<32, char>, unsigned int)">;
}

let Features = "avx512bitalg,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vpshufbitqmb512_mask : X86Builtin<"unsigned long long int(_Vector<64, char>, _Vector<64, char>, unsigned long long int)">;
}

let Features = "avx512bw,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def pmulhrsw512 : X86Builtin<"_Vector<32, short>(_Vector<32, short>, _Vector<32, short>)">;
}

let Features = "avx512bw,evex512", Attributes = [NoThrow, Const, Constexpr, RequiredVectorWidth<512>] in {
  def pmulhuw512 : X86Builtin<"_Vector<32, unsigned short>(_Vector<32, unsigned short>, _Vector<32, unsigned short>)">;
  def pmulhw512 : X86Builtin<"_Vector<32, short>(_Vector<32, short>, _Vector<32, short>)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def addpd512 : X86Builtin<"_Vector<8, double>(_Vector<8, double>, _Vector<8, double>, _Constant int)">;
  def addps512 : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Vector<16, float>, _Constant int)">;
  def divpd512 : X86Builtin<"_Vector<8, double>(_Vector<8, double>, _Vector<8, double>, _Constant int)">;
  def divps512 : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Vector<16, float>, _Constant int)">;
  def mulpd512 : X86Builtin<"_Vector<8, double>(_Vector<8, double>, _Vector<8, double>, _Constant int)">;
  def mulps512 : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Vector<16, float>, _Constant int)">;
  def subpd512 : X86Builtin<"_Vector<8, double>(_Vector<8, double>, _Vector<8, double>, _Constant int)">;
  def subps512 : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Vector<16, float>, _Constant int)">;
}

let Features = "avx512bw,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def pmaddubsw512 : X86Builtin<"_Vector<32, short>(_Vector<64, char>, _Vector<64, char>)">;
  def pmaddwd512 : X86Builtin<"_Vector<16, int>(_Vector<32, short>, _Vector<32, short>)">;
}

let Features = "avx512f", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def addss_round_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>, unsigned char, _Constant int)">;
  def divss_round_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>, unsigned char, _Constant int)">;
  def mulss_round_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>, unsigned char, _Constant int)">;
  def subss_round_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>, unsigned char, _Constant int)">;
  def maxss_round_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>, unsigned char, _Constant int)">;
  def minss_round_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>, unsigned char, _Constant int)">;
  def addsd_round_mask : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Vector<2, double>, unsigned char, _Constant int)">;
  def divsd_round_mask : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Vector<2, double>, unsigned char, _Constant int)">;
  def mulsd_round_mask : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Vector<2, double>, unsigned char, _Constant int)">;
  def subsd_round_mask : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Vector<2, double>, unsigned char, _Constant int)">;
  def maxsd_round_mask : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Vector<2, double>, unsigned char, _Constant int)">;
  def minsd_round_mask : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Vector<2, double>, unsigned char, _Constant int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def compressdf128_mask : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def compressdf256_mask : X86Builtin<"_Vector<4, double>(_Vector<4, double>, _Vector<4, double>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def compressdi128_mask : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, _Vector<2, long long int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def compressdi256_mask : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, _Vector<4, long long int>, unsigned char)">;
}

let Features = "avx512vl,avx512vbmi2", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def compresshi128_mask : X86Builtin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>, unsigned char)">;
}

let Features = "avx512vl,avx512vbmi2", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def compresshi256_mask : X86Builtin<"_Vector<16, short>(_Vector<16, short>, _Vector<16, short>, unsigned short)">;
}

let Features = "avx512vl,avx512vbmi2", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def compressqi128_mask : X86Builtin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>, unsigned short)">;
}

let Features = "avx512vl,avx512vbmi2", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def compressqi256_mask : X86Builtin<"_Vector<32, char>(_Vector<32, char>, _Vector<32, char>, unsigned int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def compresssf128_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def compresssf256_mask : X86Builtin<"_Vector<8, float>(_Vector<8, float>, _Vector<8, float>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def compresssi128_mask : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def compresssi256_mask : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<8, int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def compressstoredf128_mask : X86Builtin<"void(_Vector<2, double *>, _Vector<2, double>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def compressstoredf256_mask : X86Builtin<"void(_Vector<4, double *>, _Vector<4, double>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def compressstoredi128_mask : X86Builtin<"void(_Vector<2, long long int *>, _Vector<2, long long int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def compressstoredi256_mask : X86Builtin<"void(_Vector<4, long long int *>, _Vector<4, long long int>, unsigned char)">;
}

let Features = "avx512vl,avx512vbmi2", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def compressstorehi128_mask : X86Builtin<"void(_Vector<8, short *>, _Vector<8, short>, unsigned char)">;
}

let Features = "avx512vl,avx512vbmi2", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def compressstorehi256_mask : X86Builtin<"void(_Vector<16, short *>, _Vector<16, short>, unsigned short)">;
}

let Features = "avx512vl,avx512vbmi2", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def compressstoreqi128_mask : X86Builtin<"void(_Vector<16, char *>, _Vector<16, char>, unsigned short)">;
}

let Features = "avx512vl,avx512vbmi2", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def compressstoreqi256_mask : X86Builtin<"void(_Vector<32, char *>, _Vector<32, char>, unsigned int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def compressstoresf128_mask : X86Builtin<"void(_Vector<4, float *>, _Vector<4, float>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def compressstoresf256_mask : X86Builtin<"void(_Vector<8, float *>, _Vector<8, float>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def compressstoresi128_mask : X86Builtin<"void(_Vector<4, int *>, _Vector<4, int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def compressstoresi256_mask : X86Builtin<"void(_Vector<8, int *>, _Vector<8, int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def cvtpd2dq128_mask : X86Builtin<"_Vector<4, int>(_Vector<2, double>, _Vector<4, int>, unsigned char)">;
  def cvtpd2ps_mask : X86Builtin<"_Vector<4, float>(_Vector<2, double>, _Vector<4, float>, unsigned char)">;
  def cvtpd2udq128_mask : X86Builtin<"_Vector<4, int>(_Vector<2, double>, _Vector<4, int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def cvtpd2udq256_mask : X86Builtin<"_Vector<4, int>(_Vector<4, double>, _Vector<4, int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def cvtps2udq128_mask : X86Builtin<"_Vector<4, int>(_Vector<4, float>, _Vector<4, int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def cvtps2udq256_mask : X86Builtin<"_Vector<8, int>(_Vector<8, float>, _Vector<8, int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def cvttpd2dq128_mask : X86Builtin<"_Vector<4, int>(_Vector<2, double>, _Vector<4, int>, unsigned char)">;
  def cvttpd2udq128_mask : X86Builtin<"_Vector<4, int>(_Vector<2, double>, _Vector<4, int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def cvttpd2udq256_mask : X86Builtin<"_Vector<4, int>(_Vector<4, double>, _Vector<4, int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def cvttps2udq128_mask : X86Builtin<"_Vector<4, int>(_Vector<4, float>, _Vector<4, int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def cvttps2udq256_mask : X86Builtin<"_Vector<8, int>(_Vector<8, float>, _Vector<8, int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def expanddf128_mask : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def expanddf256_mask : X86Builtin<"_Vector<4, double>(_Vector<4, double>, _Vector<4, double>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def expanddi128_mask : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, _Vector<2, long long int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def expanddi256_mask : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, _Vector<4, long long int>, unsigned char)">;
}

let Features = "avx512vl,avx512vbmi2", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def expandhi128_mask : X86Builtin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>, unsigned char)">;
}

let Features = "avx512vl,avx512vbmi2", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def expandhi256_mask : X86Builtin<"_Vector<16, short>(_Vector<16, short>, _Vector<16, short>, unsigned short)">;
}

let Features = "avx512vl,avx512vbmi2", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def expandqi128_mask : X86Builtin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>, unsigned short)">;
}

let Features = "avx512vl,avx512vbmi2", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def expandqi256_mask : X86Builtin<"_Vector<32, char>(_Vector<32, char>, _Vector<32, char>, unsigned int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def expandloaddf128_mask : X86Builtin<"_Vector<2, double>(_Vector<2, double const *>, _Vector<2, double>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def expandloaddf256_mask : X86Builtin<"_Vector<4, double>(_Vector<4, double const *>, _Vector<4, double>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def expandloaddi128_mask : X86Builtin<"_Vector<4, int>(_Vector<2, long long int const *>, _Vector<2, long long int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def expandloaddi256_mask : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int const *>, _Vector<4, long long int>, unsigned char)">;
}

let Features = "avx512vl,avx512vbmi2", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def expandloadhi128_mask : X86Builtin<"_Vector<8, short>(_Vector<8, short const *>, _Vector<8, short>, unsigned char)">;
}

let Features = "avx512vl,avx512vbmi2", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def expandloadhi256_mask : X86Builtin<"_Vector<16, short>(_Vector<16, short const *>, _Vector<16, short>, unsigned short)">;
}

let Features = "avx512vl,avx512vbmi2", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def expandloadqi128_mask : X86Builtin<"_Vector<16, char>(_Vector<16, char const *>, _Vector<16, char>, unsigned short)">;
}

let Features = "avx512vl,avx512vbmi2", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def expandloadqi256_mask : X86Builtin<"_Vector<32, char>(_Vector<32, char const *>, _Vector<32, char>, unsigned int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def expandloadsf128_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float const *>, _Vector<4, float>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def expandloadsf256_mask : X86Builtin<"_Vector<8, float>(_Vector<8, float const *>, _Vector<8, float>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def expandloadsi128_mask : X86Builtin<"_Vector<4, int>(_Vector<4, int const *>, _Vector<4, int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def expandloadsi256_mask : X86Builtin<"_Vector<8, int>(_Vector<8, int const *>, _Vector<8, int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def expandsf128_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def expandsf256_mask : X86Builtin<"_Vector<8, float>(_Vector<8, float>, _Vector<8, float>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def expandsi128_mask : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def expandsi256_mask : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<8, int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def getexppd128_mask : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def getexppd256_mask : X86Builtin<"_Vector<4, double>(_Vector<4, double>, _Vector<4, double>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def getexpps128_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def getexpps256_mask : X86Builtin<"_Vector<8, float>(_Vector<8, float>, _Vector<8, float>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def rndscalepd_128_mask : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Constant int, _Vector<2, double>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def rndscalepd_256_mask : X86Builtin<"_Vector<4, double>(_Vector<4, double>, _Constant int, _Vector<4, double>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def rndscaleps_128_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Constant int, _Vector<4, float>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def rndscaleps_256_mask : X86Builtin<"_Vector<8, float>(_Vector<8, float>, _Constant int, _Vector<8, float>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def scalefpd128_mask : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Vector<2, double>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def scalefpd256_mask : X86Builtin<"_Vector<4, double>(_Vector<4, double>, _Vector<4, double>, _Vector<4, double>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def scalefps128_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def scalefps256_mask : X86Builtin<"_Vector<8, float>(_Vector<8, float>, _Vector<8, float>, _Vector<8, float>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def scatterdiv2df : X86Builtin<"void(void *, unsigned char, _Vector<2, long long int>, _Vector<2, double>, _Constant int)">;
  def scatterdiv2di : X86Builtin<"void(void *, unsigned char, _Vector<2, long long int>, _Vector<2, long long int>, _Constant int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def scatterdiv4df : X86Builtin<"void(void *, unsigned char, _Vector<4, long long int>, _Vector<4, double>, _Constant int)">;
  def scatterdiv4di : X86Builtin<"void(void *, unsigned char, _Vector<4, long long int>, _Vector<4, long long int>, _Constant int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def scatterdiv4sf : X86Builtin<"void(void *, unsigned char, _Vector<2, long long int>, _Vector<4, float>, _Constant int)">;
  def scatterdiv4si : X86Builtin<"void(void *, unsigned char, _Vector<2, long long int>, _Vector<4, int>, _Constant int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def scatterdiv8sf : X86Builtin<"void(void *, unsigned char, _Vector<4, long long int>, _Vector<4, float>, _Constant int)">;
  def scatterdiv8si : X86Builtin<"void(void *, unsigned char, _Vector<4, long long int>, _Vector<4, int>, _Constant int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def scattersiv2df : X86Builtin<"void(void *, unsigned char, _Vector<4, int>, _Vector<2, double>, _Constant int)">;
  def scattersiv2di : X86Builtin<"void(void *, unsigned char, _Vector<4, int>, _Vector<2, long long int>, _Constant int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def scattersiv4df : X86Builtin<"void(void *, unsigned char, _Vector<4, int>, _Vector<4, double>, _Constant int)">;
  def scattersiv4di : X86Builtin<"void(void *, unsigned char, _Vector<4, int>, _Vector<4, long long int>, _Constant int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def scattersiv4sf : X86Builtin<"void(void *, unsigned char, _Vector<4, int>, _Vector<4, float>, _Constant int)">;
  def scattersiv4si : X86Builtin<"void(void *, unsigned char, _Vector<4, int>, _Vector<4, int>, _Constant int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def scattersiv8sf : X86Builtin<"void(void *, unsigned char, _Vector<8, int>, _Vector<8, float>, _Constant int)">;
  def scattersiv8si : X86Builtin<"void(void *, unsigned char, _Vector<8, int>, _Vector<8, int>, _Constant int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vpermi2vard128 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>, _Vector<4, int>)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vpermi2vard256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<8, int>, _Vector<8, int>)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vpermi2vard512 : X86Builtin<"_Vector<16, int>(_Vector<16, int>, _Vector<16, int>, _Vector<16, int>)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vpermi2varpd128 : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, long long int>, _Vector<2, double>)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vpermi2varpd256 : X86Builtin<"_Vector<4, double>(_Vector<4, double>, _Vector<4, long long int>, _Vector<4, double>)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vpermi2varpd512 : X86Builtin<"_Vector<8, double>(_Vector<8, double>, _Vector<8, long long int>, _Vector<8, double>)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vpermi2varps128 : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, int>, _Vector<4, float>)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vpermi2varps256 : X86Builtin<"_Vector<8, float>(_Vector<8, float>, _Vector<8, int>, _Vector<8, float>)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vpermi2varps512 : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Vector<16, int>, _Vector<16, float>)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vpermi2varq128 : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, _Vector<2, long long int>, _Vector<2, long long int>)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vpermi2varq256 : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, _Vector<4, long long int>, _Vector<4, long long int>)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vpermi2varq512 : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int>, _Vector<8, long long int>, _Vector<8, long long int>)">;
}

let Features = "avx512vbmi,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vpermi2varqi128 : X86Builtin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>)">;
}

let Features = "avx512vbmi,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vpermi2varqi256 : X86Builtin<"_Vector<32, char>(_Vector<32, char>, _Vector<32, char>, _Vector<32, char>)">;
}

let Features = "avx512vbmi,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vpermi2varqi512 : X86Builtin<"_Vector<64, char>(_Vector<64, char>, _Vector<64, char>, _Vector<64, char>)">;
}

let Features = "avx512vl,avx512bw", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vpermi2varhi128 : X86Builtin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>, _Vector<8, short>)">;
}

let Features = "avx512vl,avx512bw", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vpermi2varhi256 : X86Builtin<"_Vector<16, short>(_Vector<16, short>, _Vector<16, short>, _Vector<16, short>)">;
}

let Features = "avx512bw,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vpermi2varhi512 : X86Builtin<"_Vector<32, short>(_Vector<32, short>, _Vector<32, short>, _Vector<32, short>)">;
}

let Features = "avx512vl,avx512vbmi2", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vpshldd128 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>, _Constant int)">;
}

let Features = "avx512vl,avx512vbmi2", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vpshldd256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<8, int>, _Constant int)">;
}

let Features = "avx512vbmi2,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vpshldd512 : X86Builtin<"_Vector<16, int>(_Vector<16, int>, _Vector<16, int>, _Constant int)">;
}

let Features = "avx512vl,avx512vbmi2", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vpshldq128 : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, _Vector<2, long long int>, _Constant int)">;
}

let Features = "avx512vl,avx512vbmi2", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vpshldq256 : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, _Vector<4, long long int>, _Constant int)">;
}

let Features = "avx512vbmi2,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vpshldq512 : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int>, _Vector<8, long long int>, _Constant int)">;
}

let Features = "avx512vl,avx512vbmi2", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vpshldw128 : X86Builtin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>, _Constant int)">;
}

let Features = "avx512vl,avx512vbmi2", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vpshldw256 : X86Builtin<"_Vector<16, short>(_Vector<16, short>, _Vector<16, short>, _Constant int)">;
}

let Features = "avx512vbmi2,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vpshldw512 : X86Builtin<"_Vector<32, short>(_Vector<32, short>, _Vector<32, short>, _Constant int)">;
}

let Features = "avx512vl,avx512vbmi2", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vpshldvd128 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>, _Vector<4, int>)">;
}

let Features = "avx512vl,avx512vbmi2", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vpshldvd256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<8, int>, _Vector<8, int>)">;
}

let Features = "avx512vbmi2,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vpshldvd512 : X86Builtin<"_Vector<16, int>(_Vector<16, int>, _Vector<16, int>, _Vector<16, int>)">;
}

let Features = "avx512vl,avx512vbmi2", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vpshldvq128 : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, _Vector<2, long long int>, _Vector<2, long long int>)">;
}

let Features = "avx512vl,avx512vbmi2", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vpshldvq256 : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, _Vector<4, long long int>, _Vector<4, long long int>)">;
}

let Features = "avx512vbmi2,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vpshldvq512 : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int>, _Vector<8, long long int>, _Vector<8, long long int>)">;
}

let Features = "avx512vl,avx512vbmi2", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vpshldvw128 : X86Builtin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>, _Vector<8, short>)">;
}

let Features = "avx512vl,avx512vbmi2", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vpshldvw256 : X86Builtin<"_Vector<16, short>(_Vector<16, short>, _Vector<16, short>, _Vector<16, short>)">;
}

let Features = "avx512vbmi2,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vpshldvw512 : X86Builtin<"_Vector<32, short>(_Vector<32, short>, _Vector<32, short>, _Vector<32, short>)">;
}

let Features = "avx512vl,avx512vbmi2", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vpshrdvd128 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>, _Vector<4, int>)">;
}

let Features = "avx512vl,avx512vbmi2", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vpshrdvd256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<8, int>, _Vector<8, int>)">;
}

let Features = "avx512vbmi2,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vpshrdvd512 : X86Builtin<"_Vector<16, int>(_Vector<16, int>, _Vector<16, int>, _Vector<16, int>)">;
}

let Features = "avx512vl,avx512vbmi2", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vpshrdvq128 : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, _Vector<2, long long int>, _Vector<2, long long int>)">;
}

let Features = "avx512vl,avx512vbmi2", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vpshrdvq256 : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, _Vector<4, long long int>, _Vector<4, long long int>)">;
}

let Features = "avx512vbmi2,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vpshrdvq512 : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int>, _Vector<8, long long int>, _Vector<8, long long int>)">;
}

let Features = "avx512vl,avx512vbmi2", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vpshrdvw128 : X86Builtin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>, _Vector<8, short>)">;
}

let Features = "avx512vl,avx512vbmi2", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vpshrdvw256 : X86Builtin<"_Vector<16, short>(_Vector<16, short>, _Vector<16, short>, _Vector<16, short>)">;
}

let Features = "avx512vbmi2,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vpshrdvw512 : X86Builtin<"_Vector<32, short>(_Vector<32, short>, _Vector<32, short>, _Vector<32, short>)">;
}

let Features = "avx512vl,avx512vbmi2", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vpshrdd128 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>, _Constant int)">;
}

let Features = "avx512vl,avx512vbmi2", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vpshrdd256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<8, int>, _Constant int)">;
}

let Features = "avx512vbmi2,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vpshrdd512 : X86Builtin<"_Vector<16, int>(_Vector<16, int>, _Vector<16, int>, _Constant int)">;
}

let Features = "avx512vl,avx512vbmi2", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vpshrdq128 : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, _Vector<2, long long int>, _Constant int)">;
}

let Features = "avx512vl,avx512vbmi2", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vpshrdq256 : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, _Vector<4, long long int>, _Constant int)">;
}

let Features = "avx512vbmi2,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vpshrdq512 : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int>, _Vector<8, long long int>, _Constant int)">;
}

let Features = "avx512vl,avx512vbmi2", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vpshrdw128 : X86Builtin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>, _Constant int)">;
}

let Features = "avx512vl,avx512vbmi2", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vpshrdw256 : X86Builtin<"_Vector<16, short>(_Vector<16, short>, _Vector<16, short>, _Constant int)">;
}

let Features = "avx512vbmi2,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vpshrdw512 : X86Builtin<"_Vector<32, short>(_Vector<32, short>, _Vector<32, short>, _Constant int)">;
}

let Features = "avx512bw,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def pmovswb512_mask : X86Builtin<"_Vector<32, char>(_Vector<32, short>, _Vector<32, char>, unsigned int)">;
  def pmovuswb512_mask : X86Builtin<"_Vector<32, char>(_Vector<32, short>, _Vector<32, char>, unsigned int)">;
  def pmovwb512_mask : X86Builtin<"_Vector<32, char>(_Vector<32, short>, _Vector<32, char>, unsigned int)">;
}

let Features = "avx512vl,avx512dq", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def cvtpd2qq128_mask : X86Builtin<"_Vector<2, long long int>(_Vector<2, double>, _Vector<2, long long int>, unsigned char)">;
}

let Features = "avx512vl,avx512dq", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def cvtpd2qq256_mask : X86Builtin<"_Vector<4, long long int>(_Vector<4, double>, _Vector<4, long long int>, unsigned char)">;
}

let Features = "avx512vl,avx512dq", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def cvtpd2uqq128_mask : X86Builtin<"_Vector<2, long long int>(_Vector<2, double>, _Vector<2, long long int>, unsigned char)">;
}

let Features = "avx512vl,avx512dq", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def cvtpd2uqq256_mask : X86Builtin<"_Vector<4, long long int>(_Vector<4, double>, _Vector<4, long long int>, unsigned char)">;
}

let Features = "avx512vl,avx512dq", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def cvtps2qq128_mask : X86Builtin<"_Vector<2, long long int>(_Vector<4, float>, _Vector<2, long long int>, unsigned char)">;
}

let Features = "avx512vl,avx512dq", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def cvtps2qq256_mask : X86Builtin<"_Vector<4, long long int>(_Vector<4, float>, _Vector<4, long long int>, unsigned char)">;
}

let Features = "avx512vl,avx512dq", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def cvtps2uqq128_mask : X86Builtin<"_Vector<2, long long int>(_Vector<4, float>, _Vector<2, long long int>, unsigned char)">;
}

let Features = "avx512vl,avx512dq", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def cvtps2uqq256_mask : X86Builtin<"_Vector<4, long long int>(_Vector<4, float>, _Vector<4, long long int>, unsigned char)">;
}

let Features = "avx512vl,avx512dq", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def cvtqq2ps128_mask : X86Builtin<"_Vector<4, float>(_Vector<2, long long int>, _Vector<4, float>, unsigned char)">;
  def cvttpd2qq128_mask : X86Builtin<"_Vector<2, long long int>(_Vector<2, double>, _Vector<2, long long int>, unsigned char)">;
}

let Features = "avx512vl,avx512dq", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def cvttpd2qq256_mask : X86Builtin<"_Vector<4, long long int>(_Vector<4, double>, _Vector<4, long long int>, unsigned char)">;
}

let Features = "avx512vl,avx512dq", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def cvttpd2uqq128_mask : X86Builtin<"_Vector<2, long long int>(_Vector<2, double>, _Vector<2, long long int>, unsigned char)">;
}

let Features = "avx512vl,avx512dq", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def cvttpd2uqq256_mask : X86Builtin<"_Vector<4, long long int>(_Vector<4, double>, _Vector<4, long long int>, unsigned char)">;
}

let Features = "avx512vl,avx512dq", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def cvttps2qq128_mask : X86Builtin<"_Vector<2, long long int>(_Vector<4, float>, _Vector<2, long long int>, unsigned char)">;
}

let Features = "avx512vl,avx512dq", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def cvttps2qq256_mask : X86Builtin<"_Vector<4, long long int>(_Vector<4, float>, _Vector<4, long long int>, unsigned char)">;
}

let Features = "avx512vl,avx512dq", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def cvttps2uqq128_mask : X86Builtin<"_Vector<2, long long int>(_Vector<4, float>, _Vector<2, long long int>, unsigned char)">;
}

let Features = "avx512vl,avx512dq", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def cvttps2uqq256_mask : X86Builtin<"_Vector<4, long long int>(_Vector<4, float>, _Vector<4, long long int>, unsigned char)">;
}

let Features = "avx512vl,avx512dq", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def cvtuqq2ps128_mask : X86Builtin<"_Vector<4, float>(_Vector<2, long long int>, _Vector<4, float>, unsigned char)">;
  def rangepd128_mask : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Constant int, _Vector<2, double>, unsigned char)">;
}

let Features = "avx512vl,avx512dq", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def rangepd256_mask : X86Builtin<"_Vector<4, double>(_Vector<4, double>, _Vector<4, double>, _Constant int, _Vector<4, double>, unsigned char)">;
}

let Features = "avx512vl,avx512dq", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def rangeps128_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Constant int, _Vector<4, float>, unsigned char)">;
}

let Features = "avx512vl,avx512dq", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def rangeps256_mask : X86Builtin<"_Vector<8, float>(_Vector<8, float>, _Vector<8, float>, _Constant int, _Vector<8, float>, unsigned char)">;
}

let Features = "avx512dq", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def rangesd128_round_mask : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Vector<2, double>, unsigned char, _Constant int, _Constant int)">;
  def rangess128_round_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>, unsigned char, _Constant int, _Constant int)">;
}

let Features = "avx512vl,avx512dq", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def reducepd128_mask : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Constant int, _Vector<2, double>, unsigned char)">;
}

let Features = "avx512vl,avx512dq", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def reducepd256_mask : X86Builtin<"_Vector<4, double>(_Vector<4, double>, _Constant int, _Vector<4, double>, unsigned char)">;
}

let Features = "avx512vl,avx512dq", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def reduceps128_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Constant int, _Vector<4, float>, unsigned char)">;
}

let Features = "avx512vl,avx512dq", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def reduceps256_mask : X86Builtin<"_Vector<8, float>(_Vector<8, float>, _Constant int, _Vector<8, float>, unsigned char)">;
}

let Features = "avx512dq", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def reducesd_mask : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Vector<2, double>, unsigned char, _Constant int, _Constant int)">;
  def reducess_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>, unsigned char, _Constant int, _Constant int)">;
}

let Features = "avx512vl,avx512bw", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def pmovswb128_mask : X86Builtin<"_Vector<16, char>(_Vector<8, short>, _Vector<16, char>, unsigned char)">;
}

let Features = "avx512vl,avx512bw", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def pmovswb256_mask : X86Builtin<"_Vector<16, char>(_Vector<16, short>, _Vector<16, char>, unsigned short)">;
}

let Features = "avx512vl,avx512bw", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def pmovuswb128_mask : X86Builtin<"_Vector<16, char>(_Vector<8, short>, _Vector<16, char>, unsigned char)">;
}

let Features = "avx512vl,avx512bw", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def pmovuswb256_mask : X86Builtin<"_Vector<16, char>(_Vector<16, short>, _Vector<16, char>, unsigned short)">;
}

let Features = "avx512vl,avx512bw", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def pmovwb128_mask : X86Builtin<"_Vector<16, char>(_Vector<8, short>, _Vector<16, char>, unsigned char)">;
}

let Features = "avx512dq,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def cvtpd2qq512_mask : X86Builtin<"_Vector<8, long long int>(_Vector<8, double>, _Vector<8, long long int>, unsigned char, _Constant int)">;
  def cvtpd2uqq512_mask : X86Builtin<"_Vector<8, long long int>(_Vector<8, double>, _Vector<8, long long int>, unsigned char, _Constant int)">;
  def cvtps2qq512_mask : X86Builtin<"_Vector<8, long long int>(_Vector<8, float>, _Vector<8, long long int>, unsigned char, _Constant int)">;
  def cvtps2uqq512_mask : X86Builtin<"_Vector<8, long long int>(_Vector<8, float>, _Vector<8, long long int>, unsigned char, _Constant int)">;
  def cvtqq2pd512_mask : X86Builtin<"_Vector<8, double>(_Vector<8, long long int>, _Vector<8, double>, unsigned char, _Constant int)">;
  def cvtqq2ps512_mask : X86Builtin<"_Vector<8, float>(_Vector<8, long long int>, _Vector<8, float>, unsigned char, _Constant int)">;
  def cvttpd2qq512_mask : X86Builtin<"_Vector<8, long long int>(_Vector<8, double>, _Vector<8, long long int>, unsigned char, _Constant int)">;
  def cvttpd2uqq512_mask : X86Builtin<"_Vector<8, long long int>(_Vector<8, double>, _Vector<8, long long int>, unsigned char, _Constant int)">;
  def cvttps2qq512_mask : X86Builtin<"_Vector<8, long long int>(_Vector<8, float>, _Vector<8, long long int>, unsigned char, _Constant int)">;
  def cvttps2uqq512_mask : X86Builtin<"_Vector<8, long long int>(_Vector<8, float>, _Vector<8, long long int>, unsigned char, _Constant int)">;
  def cvtuqq2pd512_mask : X86Builtin<"_Vector<8, double>(_Vector<8, long long int>, _Vector<8, double>, unsigned char, _Constant int)">;
  def cvtuqq2ps512_mask : X86Builtin<"_Vector<8, float>(_Vector<8, long long int>, _Vector<8, float>, unsigned char, _Constant int)">;
  def rangepd512_mask : X86Builtin<"_Vector<8, double>(_Vector<8, double>, _Vector<8, double>, _Constant int, _Vector<8, double>, unsigned char, _Constant int)">;
  def rangeps512_mask : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Vector<16, float>, _Constant int, _Vector<16, float>, unsigned short, _Constant int)">;
  def reducepd512_mask : X86Builtin<"_Vector<8, double>(_Vector<8, double>, _Constant int, _Vector<8, double>, unsigned char, _Constant int)">;
  def reduceps512_mask : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Constant int, _Vector<16, float>, unsigned short, _Constant int)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def prold512 : X86Builtin<"_Vector<16, int>(_Vector<16, int>, _Constant int)">;
  def prolq512 : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int>, _Constant int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def prold128 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Constant int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def prold256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Constant int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def prolq128 : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, _Constant int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def prolq256 : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, _Constant int)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def prolvd512 : X86Builtin<"_Vector<16, int>(_Vector<16, int>, _Vector<16, int>)">;
  def prolvq512 : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int>, _Vector<8, long long int>)">;
  def prord512 : X86Builtin<"_Vector<16, int>(_Vector<16, int>, _Constant int)">;
  def prorq512 : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int>, _Constant int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def prolvd128 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def prolvd256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<8, int>)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def prolvq128 : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, _Vector<2, long long int>)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def prolvq256 : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, _Vector<4, long long int>)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def prord128 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Constant int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def prord256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Constant int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def prorq128 : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, _Constant int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def prorq256 : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, _Constant int)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def prorvd512 : X86Builtin<"_Vector<16, int>(_Vector<16, int>, _Vector<16, int>)">;
  def prorvq512 : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int>, _Vector<8, long long int>)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def prorvd128 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def prorvd256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<8, int>)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def prorvq128 : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, _Vector<2, long long int>)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def prorvq256 : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, _Vector<4, long long int>)">;
}

let Features = "avx512bw,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def pshufhw512 : X86Builtin<"_Vector<32, short>(_Vector<32, short>, _Constant int)">;
  def pshuflw512 : X86Builtin<"_Vector<32, short>(_Vector<32, short>, _Constant int)">;
  def psllv32hi : X86Builtin<"_Vector<32, short>(_Vector<32, short>, _Vector<32, short>)">;
  def psllw512 : X86Builtin<"_Vector<32, short>(_Vector<32, short>, _Vector<8, short>)">;
  def psllwi512 : X86Builtin<"_Vector<32, short>(_Vector<32, short>, int)">;
}

let Features = "avx512bw,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def psllv16hi : X86Builtin<"_Vector<16, short>(_Vector<16, short>, _Vector<16, short>)">;
}

let Features = "avx512bw,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def psllv8hi : X86Builtin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def pslldi512 : X86Builtin<"_Vector<16, int>(_Vector<16, int>, int)">;
  def psllqi512 : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int>, int)">;
}

let Features = "avx512bw,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def psrlv32hi : X86Builtin<"_Vector<32, short>(_Vector<32, short>, _Vector<32, short>)">;
}

let Features = "avx512bw,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def psrlv16hi : X86Builtin<"_Vector<16, short>(_Vector<16, short>, _Vector<16, short>)">;
}

let Features = "avx512bw,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def psrlv8hi : X86Builtin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def psrldi512 : X86Builtin<"_Vector<16, int>(_Vector<16, int>, int)">;
  def psrlqi512 : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int>, int)">;
}

let Features = "avx512bw,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def psrav32hi : X86Builtin<"_Vector<32, short>(_Vector<32, short>, _Vector<32, short>)">;
}

let Features = "avx512bw,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def psrav16hi : X86Builtin<"_Vector<16, short>(_Vector<16, short>, _Vector<16, short>)">;
}

let Features = "avx512bw,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def psrav8hi : X86Builtin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def psravq128 : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, _Vector<2, long long int>)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def psravq256 : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, _Vector<4, long long int>)">;
}

let Features = "avx512bw,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def psraw512 : X86Builtin<"_Vector<32, short>(_Vector<32, short>, _Vector<8, short>)">;
  def psrawi512 : X86Builtin<"_Vector<32, short>(_Vector<32, short>, int)">;
  def psrlw512 : X86Builtin<"_Vector<32, short>(_Vector<32, short>, _Vector<8, short>)">;
  def psrlwi512 : X86Builtin<"_Vector<32, short>(_Vector<32, short>, int)">;
  def pslldqi512_byteshift : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int>, _Constant int)">;
  def psrldqi512_byteshift : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int>, _Constant int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def movdqa32load128_mask : X86Builtin<"_Vector<4, int>(_Vector<4, int const *>, _Vector<4, int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def movdqa32load256_mask : X86Builtin<"_Vector<8, int>(_Vector<8, int const *>, _Vector<8, int>, unsigned char)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def movdqa32load512_mask : X86Builtin<"_Vector<16, int>(_Vector<16, int const *>, _Vector<16, int>, unsigned short)">;
  def movdqa32store512_mask : X86Builtin<"void(_Vector<16, int *>, _Vector<16, int>, unsigned short)">;
  def movdqa64load512_mask : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int const *>, _Vector<8, long long int>, unsigned char)">;
  def movdqa64store512_mask : X86Builtin<"void(_Vector<8, long long int *>, _Vector<8, long long int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def movdqa32store128_mask : X86Builtin<"void(_Vector<4, int *>, _Vector<4, int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def movdqa32store256_mask : X86Builtin<"void(_Vector<8, int *>, _Vector<8, int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def movdqa64load128_mask : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int const *>, _Vector<2, long long int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def movdqa64load256_mask : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int const *>, _Vector<4, long long int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def movdqa64store128_mask : X86Builtin<"void(_Vector<2, long long int *>, _Vector<2, long long int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def movdqa64store256_mask : X86Builtin<"void(_Vector<4, long long int *>, _Vector<4, long long int>, unsigned char)">;
}

let Features = "avx512ifma,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vpmadd52huq512 : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int>, _Vector<8, long long int>, _Vector<8, long long int>)">;
  def vpmadd52luq512 : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int>, _Vector<8, long long int>, _Vector<8, long long int>)">;
}

let Features = "avx512ifma,avx512vl|avxifma", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vpmadd52huq128 : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, _Vector<2, long long int>, _Vector<2, long long int>)">;
}

let Features = "avx512ifma,avx512vl|avxifma", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vpmadd52huq256 : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, _Vector<4, long long int>, _Vector<4, long long int>)">;
}

let Features = "avx512ifma,avx512vl|avxifma", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vpmadd52luq128 : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, _Vector<2, long long int>, _Vector<2, long long int>)">;
}

let Features = "avx512ifma,avx512vl|avxifma", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vpmadd52luq256 : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, _Vector<4, long long int>, _Vector<4, long long int>)">;
}

let Features = "avx512f", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vcomisd : X86Builtin<"int(_Vector<2, double>, _Vector<2, double>, _Constant int, _Constant int)">;
  def vcomiss : X86Builtin<"int(_Vector<4, float>, _Vector<4, float>, _Constant int, _Constant int)">;
}

let Features = "avx512bw", Attributes = [NoThrow, Const] in {
  def kunpckdi : X86Builtin<"unsigned long long int(unsigned long long int, unsigned long long int)">;
  def kunpcksi : X86Builtin<"unsigned int(unsigned int, unsigned int)">;
}

let Features = "avx512bw,evex512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def loaddquhi512_mask : X86Builtin<"_Vector<32, short>(_Vector<32, short const *>, _Vector<32, short>, unsigned int)">;
  def loaddquqi512_mask : X86Builtin<"_Vector<64, char>(_Vector<64, char const *>, _Vector<64, char>, unsigned long long int)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def fixupimmpd512_mask : X86Builtin<"_Vector<8, double>(_Vector<8, double>, _Vector<8, double>, _Vector<8, long long int>, _Constant int, unsigned char, _Constant int)">;
  def fixupimmpd512_maskz : X86Builtin<"_Vector<8, double>(_Vector<8, double>, _Vector<8, double>, _Vector<8, long long int>, _Constant int, unsigned char, _Constant int)">;
  def fixupimmps512_mask : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Vector<16, float>, _Vector<16, int>, _Constant int, unsigned short, _Constant int)">;
  def fixupimmps512_maskz : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Vector<16, float>, _Vector<16, int>, _Constant int, unsigned short, _Constant int)">;
}

let Features = "avx512f", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def fixupimmsd_mask : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Vector<2, long long int>, _Constant int, unsigned char, _Constant int)">;
  def fixupimmsd_maskz : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Vector<2, long long int>, _Constant int, unsigned char, _Constant int)">;
  def fixupimmss_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, int>, _Constant int, unsigned char, _Constant int)">;
  def fixupimmss_maskz : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, int>, _Constant int, unsigned char, _Constant int)">;
  def getexpsd128_round_mask : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Vector<2, double>, unsigned char, _Constant int)">;
  def getexpss128_round_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>, unsigned char, _Constant int)">;
  def getmantsd_round_mask : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Constant int, _Vector<2, double>, unsigned char, _Constant int)">;
  def getmantss_round_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Constant int, _Vector<4, float>, unsigned char, _Constant int)">;
}

let Features = "avx512bw,avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def loaddquhi128_mask : X86Builtin<"_Vector<8, short>(_Vector<8, short const *>, _Vector<8, short>, unsigned char)">;
}

let Features = "avx512bw,avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def loaddquhi256_mask : X86Builtin<"_Vector<16, short>(_Vector<16, short const *>, _Vector<16, short>, unsigned short)">;
}

let Features = "avx512bw,avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def loaddquqi128_mask : X86Builtin<"_Vector<16, char>(_Vector<16, char const *>, _Vector<16, char>, unsigned short)">;
}

let Features = "avx512bw,avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def loaddquqi256_mask : X86Builtin<"_Vector<32, char>(_Vector<32, char const *>, _Vector<32, char>, unsigned int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def fixupimmpd128_mask : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Vector<2, long long int>, _Constant int, unsigned char)">;
  def fixupimmpd128_maskz : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Vector<2, long long int>, _Constant int, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def fixupimmpd256_mask : X86Builtin<"_Vector<4, double>(_Vector<4, double>, _Vector<4, double>, _Vector<4, long long int>, _Constant int, unsigned char)">;
  def fixupimmpd256_maskz : X86Builtin<"_Vector<4, double>(_Vector<4, double>, _Vector<4, double>, _Vector<4, long long int>, _Constant int, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def fixupimmps128_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, int>, _Constant int, unsigned char)">;
  def fixupimmps128_maskz : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, int>, _Constant int, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def fixupimmps256_mask : X86Builtin<"_Vector<8, float>(_Vector<8, float>, _Vector<8, float>, _Vector<8, int>, _Constant int, unsigned char)">;
  def fixupimmps256_maskz : X86Builtin<"_Vector<8, float>(_Vector<8, float>, _Vector<8, float>, _Vector<8, int>, _Constant int, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def loadapd128_mask : X86Builtin<"_Vector<2, double>(_Vector<2, double const *>, _Vector<2, double>, unsigned char)">;
}

let Features = "avx512f", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def loadsd128_mask : X86Builtin<"_Vector<2, double>(_Vector<2, double const *>, _Vector<2, double>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def loadapd256_mask : X86Builtin<"_Vector<4, double>(_Vector<4, double const *>, _Vector<4, double>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def loadaps128_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float const *>, _Vector<4, float>, unsigned char)">;
}

let Features = "avx512f", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def loadss128_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float const *>, _Vector<4, float>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def loadaps256_mask : X86Builtin<"_Vector<8, float>(_Vector<8, float const *>, _Vector<8, float>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def loaddqudi128_mask : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int const *>, _Vector<2, long long int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def loaddqudi256_mask : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int const *>, _Vector<4, long long int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def loaddqusi128_mask : X86Builtin<"_Vector<4, int>(_Vector<4, int const *>, _Vector<4, int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def loaddqusi256_mask : X86Builtin<"_Vector<8, int>(_Vector<8, int const *>, _Vector<8, int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def loadupd128_mask : X86Builtin<"_Vector<2, double>(_Vector<2, double const *>, _Vector<2, double>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def loadupd256_mask : X86Builtin<"_Vector<4, double>(_Vector<4, double const *>, _Vector<4, double>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def loadups128_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float const *>, _Vector<4, float>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def loadups256_mask : X86Builtin<"_Vector<8, float>(_Vector<8, float const *>, _Vector<8, float>, unsigned char)">;
}

let Features = "avx512bw,evex512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def storedquhi512_mask : X86Builtin<"void(_Vector<32, short *>, _Vector<32, short>, unsigned int)">;
  def storedquqi512_mask : X86Builtin<"void(_Vector<64, char *>, _Vector<64, char>, unsigned long long int)">;
}

let Features = "avx512vl,avx512bw", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def storedquhi128_mask : X86Builtin<"void(_Vector<8, short *>, _Vector<8, short>, unsigned char)">;
}

let Features = "avx512vl,avx512bw", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def storedquhi256_mask : X86Builtin<"void(_Vector<16, short *>, _Vector<16, short>, unsigned short)">;
}

let Features = "avx512vl,avx512bw", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def storedquqi128_mask : X86Builtin<"void(_Vector<16, char *>, _Vector<16, char>, unsigned short)">;
}

let Features = "avx512vl,avx512bw", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def storedquqi256_mask : X86Builtin<"void(_Vector<32, char *>, _Vector<32, char>, unsigned int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def storeapd128_mask : X86Builtin<"void(_Vector<2, double *>, _Vector<2, double>, unsigned char)">;
}

let Features = "avx512f", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def storesd128_mask : X86Builtin<"void(_Vector<2, double *>, _Vector<2, double>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def storeapd256_mask : X86Builtin<"void(_Vector<4, double *>, _Vector<4, double>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def storeaps128_mask : X86Builtin<"void(_Vector<4, float *>, _Vector<4, float>, unsigned char)">;
}

let Features = "avx512f", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def storess128_mask : X86Builtin<"void(_Vector<4, float *>, _Vector<4, float>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def storeaps256_mask : X86Builtin<"void(_Vector<8, float *>, _Vector<8, float>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def storedqudi128_mask : X86Builtin<"void(_Vector<2, long long int *>, _Vector<2, long long int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def storedqudi256_mask : X86Builtin<"void(_Vector<4, long long int *>, _Vector<4, long long int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def storedqusi128_mask : X86Builtin<"void(_Vector<4, int *>, _Vector<4, int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def storedqusi256_mask : X86Builtin<"void(_Vector<8, int *>, _Vector<8, int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def storeupd128_mask : X86Builtin<"void(_Vector<2, double *>, _Vector<2, double>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def storeupd256_mask : X86Builtin<"void(_Vector<4, double *>, _Vector<4, double>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def storeups128_mask : X86Builtin<"void(_Vector<4, float *>, _Vector<4, float>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def storeups256_mask : X86Builtin<"void(_Vector<8, float *>, _Vector<8, float>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def rcp14pd128_mask : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def rcp14pd256_mask : X86Builtin<"_Vector<4, double>(_Vector<4, double>, _Vector<4, double>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def rcp14ps128_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def rcp14ps256_mask : X86Builtin<"_Vector<8, float>(_Vector<8, float>, _Vector<8, float>, unsigned char)">;
}

let Features = "avx512cd,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vplzcntd_128 : X86Builtin<"_Vector<4, int>(_Vector<4, int>)">;
}

let Features = "avx512cd,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vplzcntd_256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>)">;
}

let Features = "avx512cd,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vplzcntq_128 : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>)">;
}

let Features = "avx512cd,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vplzcntq_256 : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>)">;
}

let Features = "avx512f", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vcvtsd2si32 : X86Builtin<"int(_Vector<2, double>, _Constant int)">;
  def vcvtsd2usi32 : X86Builtin<"unsigned int(_Vector<2, double>, _Constant int)">;
  def vcvtss2si32 : X86Builtin<"int(_Vector<4, float>, _Constant int)">;
  def vcvtss2usi32 : X86Builtin<"unsigned int(_Vector<4, float>, _Constant int)">;
  def vcvttsd2si32 : X86Builtin<"int(_Vector<2, double>, _Constant int)">;
  def vcvttsd2usi32 : X86Builtin<"unsigned int(_Vector<2, double>, _Constant int)">;
  def vcvttss2si32 : X86Builtin<"int(_Vector<4, float>, _Constant int)">;
  def vcvttss2usi32 : X86Builtin<"unsigned int(_Vector<4, float>, _Constant int)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vpermilpd512 : X86Builtin<"_Vector<8, double>(_Vector<8, double>, _Constant int)">;
  def vpermilps512 : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Constant int)">;
  def vpermilvarpd512 : X86Builtin<"_Vector<8, double>(_Vector<8, double>, _Vector<8, long long int>)">;
  def vpermilvarps512 : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Vector<16, int>)">;
}

let Features = "avx512f", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def rndscalesd_round_mask : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Vector<2, double>, unsigned char, _Constant int, _Constant int)">;
  def rndscaless_round_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>, unsigned char, _Constant int, _Constant int)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def scalefpd512_mask : X86Builtin<"_Vector<8, double>(_Vector<8, double>, _Vector<8, double>, _Vector<8, double>, unsigned char, _Constant int)">;
  def scalefps512_mask : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Vector<16, float>, _Vector<16, float>, unsigned short, _Constant int)">;
}

let Features = "avx512f", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def scalefsd_round_mask : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Vector<2, double>, unsigned char, _Constant int)">;
  def scalefss_round_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>, unsigned char, _Constant int)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def psradi512 : X86Builtin<"_Vector<16, int>(_Vector<16, int>, int)">;
  def psraqi512 : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int>, int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def psraq128 : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, _Vector<2, long long int>)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def psraq256 : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, _Vector<2, long long int>)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def psraqi128 : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def psraqi256 : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, int)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def pslld512 : X86Builtin<"_Vector<16, int>(_Vector<16, int>, _Vector<4, int>)">;
  def psllq512 : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int>, _Vector<2, long long int>)">;
  def psllv16si : X86Builtin<"_Vector<16, int>(_Vector<16, int>, _Vector<16, int>)">;
  def psllv8di : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int>, _Vector<8, long long int>)">;
  def psrad512 : X86Builtin<"_Vector<16, int>(_Vector<16, int>, _Vector<4, int>)">;
  def psraq512 : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int>, _Vector<2, long long int>)">;
  def psrav16si : X86Builtin<"_Vector<16, int>(_Vector<16, int>, _Vector<16, int>)">;
  def psrav8di : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int>, _Vector<8, long long int>)">;
  def psrld512 : X86Builtin<"_Vector<16, int>(_Vector<16, int>, _Vector<4, int>)">;
  def psrlq512 : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int>, _Vector<2, long long int>)">;
  def psrlv16si : X86Builtin<"_Vector<16, int>(_Vector<16, int>, _Vector<16, int>)">;
  def psrlv8di : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int>, _Vector<8, long long int>)">;
  def pternlogd512_mask : X86Builtin<"_Vector<16, int>(_Vector<16, int>, _Vector<16, int>, _Vector<16, int>, _Constant int, unsigned short)">;
  def pternlogd512_maskz : X86Builtin<"_Vector<16, int>(_Vector<16, int>, _Vector<16, int>, _Vector<16, int>, _Constant int, unsigned short)">;
  def pternlogq512_mask : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int>, _Vector<8, long long int>, _Vector<8, long long int>, _Constant int, unsigned char)">;
  def pternlogq512_maskz : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int>, _Vector<8, long long int>, _Vector<8, long long int>, _Constant int, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def pternlogd128_mask : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>, _Vector<4, int>, _Constant int, unsigned char)">;
  def pternlogd128_maskz : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>, _Vector<4, int>, _Constant int, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def pternlogd256_mask : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<8, int>, _Vector<8, int>, _Constant int, unsigned char)">;
  def pternlogd256_maskz : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<8, int>, _Vector<8, int>, _Constant int, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def pternlogq128_mask : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, _Vector<2, long long int>, _Vector<2, long long int>, _Constant int, unsigned char)">;
  def pternlogq128_maskz : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, _Vector<2, long long int>, _Vector<2, long long int>, _Constant int, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def pternlogq256_mask : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, _Vector<4, long long int>, _Vector<4, long long int>, _Constant int, unsigned char)">;
  def pternlogq256_maskz : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, _Vector<4, long long int>, _Vector<4, long long int>, _Constant int, unsigned char)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def shuf_f32x4 : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Vector<16, float>, _Constant int)">;
  def shuf_f64x2 : X86Builtin<"_Vector<8, double>(_Vector<8, double>, _Vector<8, double>, _Constant int)">;
  def shuf_i32x4 : X86Builtin<"_Vector<16, int>(_Vector<16, int>, _Vector<16, int>, _Constant int)">;
  def shuf_i64x2 : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int>, _Vector<8, long long int>, _Constant int)">;
  def shufpd512 : X86Builtin<"_Vector<8, double>(_Vector<8, double>, _Vector<8, double>, _Constant int)">;
  def shufps512 : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Vector<16, float>, _Constant int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def shuf_f32x4_256 : X86Builtin<"_Vector<8, float>(_Vector<8, float>, _Vector<8, float>, _Constant int)">;
  def shuf_f64x2_256 : X86Builtin<"_Vector<4, double>(_Vector<4, double>, _Vector<4, double>, _Constant int)">;
  def shuf_i32x4_256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<8, int>, _Constant int)">;
  def shuf_i64x2_256 : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, _Vector<4, long long int>, _Constant int)">;
}

let Features = "avx512f", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def sqrtsd_round_mask : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Vector<2, double>, unsigned char, _Constant int)">;
  def sqrtss_round_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>, unsigned char, _Constant int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def rsqrt14pd128_mask : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def rsqrt14pd256_mask : X86Builtin<"_Vector<4, double>(_Vector<4, double>, _Vector<4, double>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def rsqrt14ps128_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def rsqrt14ps256_mask : X86Builtin<"_Vector<8, float>(_Vector<8, float>, _Vector<8, float>, unsigned char)">;
}

let Features = "avx512bw,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def cvtb2mask512 : X86Builtin<"unsigned long long int(_Vector<64, char>)">;
  def cvtmask2b512 : X86Builtin<"_Vector<64, char>(unsigned long long int)">;
  def cvtmask2w512 : X86Builtin<"_Vector<32, short>(unsigned int)">;
}

let Features = "avx512dq,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def cvtd2mask512 : X86Builtin<"unsigned short(_Vector<16, int>)">;
  def cvtmask2d512 : X86Builtin<"_Vector<16, int>(unsigned short)">;
  def cvtmask2q512 : X86Builtin<"_Vector<8, long long int>(unsigned char)">;
  def cvtq2mask512 : X86Builtin<"unsigned char(_Vector<8, long long int>)">;
}

let Features = "avx512bw,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def cvtb2mask128 : X86Builtin<"unsigned short(_Vector<16, char>)">;
}

let Features = "avx512bw,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def cvtb2mask256 : X86Builtin<"unsigned int(_Vector<32, char>)">;
}

let Features = "avx512bw,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def cvtmask2b128 : X86Builtin<"_Vector<16, char>(unsigned short)">;
}

let Features = "avx512bw,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def cvtmask2b256 : X86Builtin<"_Vector<32, char>(unsigned int)">;
}

let Features = "avx512bw,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def cvtmask2w128 : X86Builtin<"_Vector<8, short>(unsigned char)">;
}

let Features = "avx512bw,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def cvtmask2w256 : X86Builtin<"_Vector<16, short>(unsigned short)">;
}

let Features = "avx512dq,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def cvtd2mask128 : X86Builtin<"unsigned char(_Vector<4, int>)">;
}

let Features = "avx512dq,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def cvtd2mask256 : X86Builtin<"unsigned char(_Vector<8, int>)">;
}

let Features = "avx512dq,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def cvtmask2d128 : X86Builtin<"_Vector<4, int>(unsigned char)">;
}

let Features = "avx512dq,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def cvtmask2d256 : X86Builtin<"_Vector<8, int>(unsigned char)">;
}

let Features = "avx512dq,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def cvtmask2q128 : X86Builtin<"_Vector<2, long long int>(unsigned char)">;
}

let Features = "avx512dq,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def cvtmask2q256 : X86Builtin<"_Vector<4, long long int>(unsigned char)">;
}

let Features = "avx512dq,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def cvtq2mask128 : X86Builtin<"unsigned char(_Vector<2, long long int>)">;
}

let Features = "avx512dq,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def cvtq2mask256 : X86Builtin<"unsigned char(_Vector<4, long long int>)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def pmovsdb512_mask : X86Builtin<"_Vector<16, char>(_Vector<16, int>, _Vector<16, char>, unsigned short)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def pmovsdb512mem_mask : X86Builtin<"void(_Vector<16, char *>, _Vector<16, int>, unsigned short)">;
}

let Features = "avx512bw,evex512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def pmovswb512mem_mask : X86Builtin<"void(_Vector<32, char *>, _Vector<32, short>, unsigned int)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def pmovsdw512_mask : X86Builtin<"_Vector<16, short>(_Vector<16, int>, _Vector<16, short>, unsigned short)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def pmovsdw512mem_mask : X86Builtin<"void(_Vector<16, short *>, _Vector<16, int>, unsigned short)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def pmovsqb512_mask : X86Builtin<"_Vector<16, char>(_Vector<8, long long int>, _Vector<16, char>, unsigned char)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def pmovsqb512mem_mask : X86Builtin<"void(_Vector<16, char *>, _Vector<8, long long int>, unsigned char)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def pmovsqd512_mask : X86Builtin<"_Vector<8, int>(_Vector<8, long long int>, _Vector<8, int>, unsigned char)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def pmovsqd512mem_mask : X86Builtin<"void(_Vector<8, int *>, _Vector<8, long long int>, unsigned char)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def pmovsqw512_mask : X86Builtin<"_Vector<8, short>(_Vector<8, long long int>, _Vector<8, short>, unsigned char)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def pmovsqw512mem_mask : X86Builtin<"void(_Vector<8, short *>, _Vector<8, long long int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def pmovsdb128_mask : X86Builtin<"_Vector<16, char>(_Vector<4, int>, _Vector<16, char>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def pmovsdb128mem_mask : X86Builtin<"void(_Vector<16, char *>, _Vector<4, int>, unsigned char)">;
}

let Features = "avx512vl,avx512bw", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def pmovswb128mem_mask : X86Builtin<"void(_Vector<16, char *>, _Vector<8, short>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def pmovsdb256_mask : X86Builtin<"_Vector<16, char>(_Vector<8, int>, _Vector<16, char>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def pmovsdb256mem_mask : X86Builtin<"void(_Vector<16, char *>, _Vector<8, int>, unsigned char)">;
}

let Features = "avx512vl,avx512bw", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def pmovswb256mem_mask : X86Builtin<"void(_Vector<16, char *>, _Vector<16, short>, unsigned short)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def pmovsdw128_mask : X86Builtin<"_Vector<8, short>(_Vector<4, int>, _Vector<8, short>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def pmovsdw128mem_mask : X86Builtin<"void(_Vector<8, short *>, _Vector<4, int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def pmovsdw256_mask : X86Builtin<"_Vector<8, short>(_Vector<8, int>, _Vector<8, short>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def pmovsdw256mem_mask : X86Builtin<"void(_Vector<8, short *>, _Vector<8, int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def pmovsqb128_mask : X86Builtin<"_Vector<16, char>(_Vector<2, long long int>, _Vector<16, char>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def pmovsqb128mem_mask : X86Builtin<"void(_Vector<16, char *>, _Vector<2, long long int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def pmovsqb256_mask : X86Builtin<"_Vector<16, char>(_Vector<4, long long int>, _Vector<16, char>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def pmovsqb256mem_mask : X86Builtin<"void(_Vector<16, char *>, _Vector<4, long long int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def pmovsqd128_mask : X86Builtin<"_Vector<4, int>(_Vector<2, long long int>, _Vector<4, int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def pmovsqd128mem_mask : X86Builtin<"void(_Vector<4, int *>, _Vector<2, long long int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def pmovsqd256_mask : X86Builtin<"_Vector<4, int>(_Vector<4, long long int>, _Vector<4, int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def pmovsqd256mem_mask : X86Builtin<"void(_Vector<4, int *>, _Vector<4, long long int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def pmovsqw128_mask : X86Builtin<"_Vector<8, short>(_Vector<2, long long int>, _Vector<8, short>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def pmovsqw128mem_mask : X86Builtin<"void(_Vector<8, short *>, _Vector<2, long long int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def pmovsqw256_mask : X86Builtin<"_Vector<8, short>(_Vector<4, long long int>, _Vector<8, short>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def pmovsqw256mem_mask : X86Builtin<"void(_Vector<8, short *>, _Vector<4, long long int>, unsigned char)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def pmovusdb512_mask : X86Builtin<"_Vector<16, char>(_Vector<16, int>, _Vector<16, char>, unsigned short)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def pmovusdb512mem_mask : X86Builtin<"void(_Vector<16, char *>, _Vector<16, int>, unsigned short)">;
}

let Features = "avx512bw,evex512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def pmovuswb512mem_mask : X86Builtin<"void(_Vector<32, char *>, _Vector<32, short>, unsigned int)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def pmovusdw512_mask : X86Builtin<"_Vector<16, short>(_Vector<16, int>, _Vector<16, short>, unsigned short)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def pmovusdw512mem_mask : X86Builtin<"void(_Vector<16, short *>, _Vector<16, int>, unsigned short)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def pmovusqb512_mask : X86Builtin<"_Vector<16, char>(_Vector<8, long long int>, _Vector<16, char>, unsigned char)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def pmovusqb512mem_mask : X86Builtin<"void(_Vector<16, char *>, _Vector<8, long long int>, unsigned char)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def pmovusqd512_mask : X86Builtin<"_Vector<8, int>(_Vector<8, long long int>, _Vector<8, int>, unsigned char)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def pmovusqd512mem_mask : X86Builtin<"void(_Vector<8, int *>, _Vector<8, long long int>, unsigned char)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def pmovusqw512_mask : X86Builtin<"_Vector<8, short>(_Vector<8, long long int>, _Vector<8, short>, unsigned char)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def pmovusqw512mem_mask : X86Builtin<"void(_Vector<8, short *>, _Vector<8, long long int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def pmovusdb128_mask : X86Builtin<"_Vector<16, char>(_Vector<4, int>, _Vector<16, char>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def pmovusdb128mem_mask : X86Builtin<"void(_Vector<16, char *>, _Vector<4, int>, unsigned char)">;
}

let Features = "avx512vl,avx512bw", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def pmovuswb128mem_mask : X86Builtin<"void(_Vector<16, char *>, _Vector<8, short>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def pmovusdb256_mask : X86Builtin<"_Vector<16, char>(_Vector<8, int>, _Vector<16, char>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def pmovusdb256mem_mask : X86Builtin<"void(_Vector<16, char *>, _Vector<8, int>, unsigned char)">;
}

let Features = "avx512vl,avx512bw", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def pmovuswb256mem_mask : X86Builtin<"void(_Vector<16, char *>, _Vector<16, short>, unsigned short)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def pmovusdw128_mask : X86Builtin<"_Vector<8, short>(_Vector<4, int>, _Vector<8, short>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def pmovusdw128mem_mask : X86Builtin<"void(_Vector<8, short *>, _Vector<4, int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def pmovusdw256_mask : X86Builtin<"_Vector<8, short>(_Vector<8, int>, _Vector<8, short>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def pmovusdw256mem_mask : X86Builtin<"void(_Vector<8, short *>, _Vector<8, int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def pmovusqb128_mask : X86Builtin<"_Vector<16, char>(_Vector<2, long long int>, _Vector<16, char>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def pmovusqb128mem_mask : X86Builtin<"void(_Vector<16, char *>, _Vector<2, long long int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def pmovusqb256_mask : X86Builtin<"_Vector<16, char>(_Vector<4, long long int>, _Vector<16, char>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def pmovusqb256mem_mask : X86Builtin<"void(_Vector<16, char *>, _Vector<4, long long int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def pmovusqd128_mask : X86Builtin<"_Vector<4, int>(_Vector<2, long long int>, _Vector<4, int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def pmovusqd128mem_mask : X86Builtin<"void(_Vector<4, int *>, _Vector<2, long long int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def pmovusqd256_mask : X86Builtin<"_Vector<4, int>(_Vector<4, long long int>, _Vector<4, int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def pmovusqd256mem_mask : X86Builtin<"void(_Vector<4, int *>, _Vector<4, long long int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def pmovusqw128_mask : X86Builtin<"_Vector<8, short>(_Vector<2, long long int>, _Vector<8, short>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def pmovusqw128mem_mask : X86Builtin<"void(_Vector<8, short *>, _Vector<2, long long int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def pmovusqw256_mask : X86Builtin<"_Vector<8, short>(_Vector<4, long long int>, _Vector<8, short>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def pmovusqw256mem_mask : X86Builtin<"void(_Vector<8, short *>, _Vector<4, long long int>, unsigned char)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def pmovdb512_mask : X86Builtin<"_Vector<16, char>(_Vector<16, int>, _Vector<16, char>, unsigned short)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def pmovdb512mem_mask : X86Builtin<"void(_Vector<16, char *>, _Vector<16, int>, unsigned short)">;
}

let Features = "avx512bw,evex512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def pmovwb512mem_mask : X86Builtin<"void(_Vector<32, char *>, _Vector<32, short>, unsigned int)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def pmovdw512_mask : X86Builtin<"_Vector<16, short>(_Vector<16, int>, _Vector<16, short>, unsigned short)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def pmovdw512mem_mask : X86Builtin<"void(_Vector<16, short *>, _Vector<16, int>, unsigned short)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def pmovqb512_mask : X86Builtin<"_Vector<16, char>(_Vector<8, long long int>, _Vector<16, char>, unsigned char)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def pmovqb512mem_mask : X86Builtin<"void(_Vector<16, char *>, _Vector<8, long long int>, unsigned char)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def pmovqd512_mask : X86Builtin<"_Vector<8, int>(_Vector<8, long long int>, _Vector<8, int>, unsigned char)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def pmovqd512mem_mask : X86Builtin<"void(_Vector<8, int *>, _Vector<8, long long int>, unsigned char)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def pmovqw512_mask : X86Builtin<"_Vector<8, short>(_Vector<8, long long int>, _Vector<8, short>, unsigned char)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def pmovqw512mem_mask : X86Builtin<"void(_Vector<8, short *>, _Vector<8, long long int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def pmovdb128_mask : X86Builtin<"_Vector<16, char>(_Vector<4, int>, _Vector<16, char>, unsigned char)">;
}

let Features = "avx512vl,avx512bw", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def pmovwb128mem_mask : X86Builtin<"void(_Vector<16, char *>, _Vector<8, short>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def pmovdb128mem_mask : X86Builtin<"void(_Vector<16, char *>, _Vector<4, int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def pmovdb256_mask : X86Builtin<"_Vector<16, char>(_Vector<8, int>, _Vector<16, char>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def pmovdb256mem_mask : X86Builtin<"void(_Vector<16, char *>, _Vector<8, int>, unsigned char)">;
}

let Features = "avx512vl,avx512bw", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def pmovwb256mem_mask : X86Builtin<"void(_Vector<16, char *>, _Vector<16, short>, unsigned short)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def pmovdw128_mask : X86Builtin<"_Vector<8, short>(_Vector<4, int>, _Vector<8, short>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def pmovdw128mem_mask : X86Builtin<"void(_Vector<8, short *>, _Vector<4, int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def pmovdw256_mask : X86Builtin<"_Vector<8, short>(_Vector<8, int>, _Vector<8, short>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def pmovdw256mem_mask : X86Builtin<"void(_Vector<8, short *>, _Vector<8, int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def pmovqb128_mask : X86Builtin<"_Vector<16, char>(_Vector<2, long long int>, _Vector<16, char>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def pmovqb128mem_mask : X86Builtin<"void(_Vector<16, char *>, _Vector<2, long long int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def pmovqb256_mask : X86Builtin<"_Vector<16, char>(_Vector<4, long long int>, _Vector<16, char>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def pmovqb256mem_mask : X86Builtin<"void(_Vector<16, char *>, _Vector<4, long long int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def pmovqd128_mask : X86Builtin<"_Vector<4, int>(_Vector<2, long long int>, _Vector<4, int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def pmovqd128mem_mask : X86Builtin<"void(_Vector<4, int *>, _Vector<2, long long int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def pmovqd256mem_mask : X86Builtin<"void(_Vector<4, int *>, _Vector<4, long long int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def pmovqw128_mask : X86Builtin<"_Vector<8, short>(_Vector<2, long long int>, _Vector<8, short>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def pmovqw128mem_mask : X86Builtin<"void(_Vector<8, short *>, _Vector<2, long long int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def pmovqw256_mask : X86Builtin<"_Vector<8, short>(_Vector<4, long long int>, _Vector<8, short>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def pmovqw256mem_mask : X86Builtin<"void(_Vector<8, short *>, _Vector<4, long long int>, unsigned char)">;
}

let Features = "avx512dq,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def extractf32x8_mask : X86Builtin<"_Vector<8, float>(_Vector<16, float>, _Constant int, _Vector<8, float>, unsigned char)">;
  def extractf64x2_512_mask : X86Builtin<"_Vector<2, double>(_Vector<8, double>, _Constant int, _Vector<2, double>, unsigned char)">;
  def extracti32x8_mask : X86Builtin<"_Vector<8, int>(_Vector<16, int>, _Constant int, _Vector<8, int>, unsigned char)">;
  def extracti64x2_512_mask : X86Builtin<"_Vector<2, long long int>(_Vector<8, long long int>, _Constant int, _Vector<2, long long int>, unsigned char)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def extracti32x4_mask : X86Builtin<"_Vector<4, int>(_Vector<16, int>, _Constant int, _Vector<4, int>, unsigned char)">;
  def extracti64x4_mask : X86Builtin<"_Vector<4, long long int>(_Vector<8, long long int>, _Constant int, _Vector<4, long long int>, unsigned char)">;
}

let Features = "avx512dq,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def extractf64x2_256_mask : X86Builtin<"_Vector<2, double>(_Vector<4, double>, _Constant int, _Vector<2, double>, unsigned char)">;
  def extracti64x2_256_mask : X86Builtin<"_Vector<2, long long int>(_Vector<4, long long int>, _Constant int, _Vector<2, long long int>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def extractf32x4_256_mask : X86Builtin<"_Vector<4, float>(_Vector<8, float>, _Constant int, _Vector<4, float>, unsigned char)">;
  def extracti32x4_256_mask : X86Builtin<"_Vector<4, int>(_Vector<8, int>, _Constant int, _Vector<4, int>, unsigned char)">;
}

let Features = "avx512dq,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def insertf32x8 : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Vector<8, float>, _Constant int)">;
  def insertf64x2_512 : X86Builtin<"_Vector<8, double>(_Vector<8, double>, _Vector<2, double>, _Constant int)">;
  def inserti32x8 : X86Builtin<"_Vector<16, int>(_Vector<16, int>, _Vector<8, int>, _Constant int)">;
  def inserti64x2_512 : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int>, _Vector<2, long long int>, _Constant int)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def insertf64x4 : X86Builtin<"_Vector<8, double>(_Vector<8, double>, _Vector<4, double>, _Constant int)">;
  def inserti64x4 : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int>, _Vector<4, long long int>, _Constant int)">;
}

let Features = "avx512dq,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def insertf64x2_256 : X86Builtin<"_Vector<4, double>(_Vector<4, double>, _Vector<2, double>, _Constant int)">;
  def inserti64x2_256 : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, _Vector<2, long long int>, _Constant int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def insertf32x4_256 : X86Builtin<"_Vector<8, float>(_Vector<8, float>, _Vector<4, float>, _Constant int)">;
  def inserti32x4_256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<4, int>, _Constant int)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def insertf32x4 : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Vector<4, float>, _Constant int)">;
  def inserti32x4 : X86Builtin<"_Vector<16, int>(_Vector<16, int>, _Vector<4, int>, _Constant int)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def getmantpd128_mask : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Constant int, _Vector<2, double>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def getmantpd256_mask : X86Builtin<"_Vector<4, double>(_Vector<4, double>, _Constant int, _Vector<4, double>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def getmantps128_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Constant int, _Vector<4, float>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def getmantps256_mask : X86Builtin<"_Vector<8, float>(_Vector<8, float>, _Constant int, _Vector<8, float>, unsigned char)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def getmantpd512_mask : X86Builtin<"_Vector<8, double>(_Vector<8, double>, _Constant int, _Vector<8, double>, unsigned char, _Constant int)">;
  def getmantps512_mask : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Constant int, _Vector<16, float>, unsigned short, _Constant int)">;
  def getexppd512_mask : X86Builtin<"_Vector<8, double>(_Vector<8, double>, _Vector<8, double>, unsigned char, _Constant int)">;
  def getexpps512_mask : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Vector<16, float>, unsigned short, _Constant int)">;
}

let Features = "avx512f", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vfmaddss3_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>, unsigned char, _Constant int)">;
  def vfmaddss3_maskz : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>, unsigned char, _Constant int)">;
  def vfmaddss3_mask3 : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>, unsigned char, _Constant int)">;
  def vfmaddsd3_mask : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Vector<2, double>, unsigned char, _Constant int)">;
  def vfmaddsd3_maskz : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Vector<2, double>, unsigned char, _Constant int)">;
  def vfmaddsd3_mask3 : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Vector<2, double>, unsigned char, _Constant int)">;
  def vfmsubsd3_mask3 : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Vector<2, double>, unsigned char, _Constant int)">;
  def vfmsubss3_mask3 : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>, unsigned char, _Constant int)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def permdf512 : X86Builtin<"_Vector<8, double>(_Vector<8, double>, _Constant int)">;
  def permdi512 : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int>, _Constant int)">;
}

let Features = "avx512bw,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def permvarhi512 : X86Builtin<"_Vector<32, short>(_Vector<32, short>, _Vector<32, short>)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def permvardf512 : X86Builtin<"_Vector<8, double>(_Vector<8, double>, _Vector<8, long long int>)">;
  def permvardi512 : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int>, _Vector<8, long long int>)">;
  def permvarsf512 : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Vector<16, int>)">;
  def permvarsi512 : X86Builtin<"_Vector<16, int>(_Vector<16, int>, _Vector<16, int>)">;
}

let Features = "avx512vbmi,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def permvarqi512 : X86Builtin<"_Vector<64, char>(_Vector<64, char>, _Vector<64, char>)">;
}

let Features = "avx512vbmi,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def permvarqi128 : X86Builtin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>)">;
}

let Features = "avx512vbmi,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def permvarqi256 : X86Builtin<"_Vector<32, char>(_Vector<32, char>, _Vector<32, char>)">;
}

let Features = "avx512bw,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def permvarhi128 : X86Builtin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>)">;
}

let Features = "avx512bw,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def permvarhi256 : X86Builtin<"_Vector<16, short>(_Vector<16, short>, _Vector<16, short>)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def permvardf256 : X86Builtin<"_Vector<4, double>(_Vector<4, double>, _Vector<4, long long int>)">;
  def permvardi256 : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, _Vector<4, long long int>)">;
}

let Features = "avx512dq,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def fpclasspd128_mask : X86Builtin<"unsigned char(_Vector<2, double>, _Constant int, unsigned char)">;
}

let Features = "avx512dq,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def fpclasspd256_mask : X86Builtin<"unsigned char(_Vector<4, double>, _Constant int, unsigned char)">;
}

let Features = "avx512dq,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def fpclassps128_mask : X86Builtin<"unsigned char(_Vector<4, float>, _Constant int, unsigned char)">;
}

let Features = "avx512dq,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def fpclassps256_mask : X86Builtin<"unsigned char(_Vector<8, float>, _Constant int, unsigned char)">;
}

let Features = "avx512dq,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def fpclassps512_mask : X86Builtin<"unsigned short(_Vector<16, float>, _Constant int, unsigned short)">;
  def fpclasspd512_mask : X86Builtin<"unsigned char(_Vector<8, double>, _Constant int, unsigned char)">;
}

let Features = "avx512dq", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def fpclasssd_mask : X86Builtin<"unsigned char(_Vector<2, double>, _Constant int, unsigned char)">;
  def fpclassss_mask : X86Builtin<"unsigned char(_Vector<4, float>, _Constant int, unsigned char)">;
}

let Features = "avx512dq", Attributes = [NoThrow, Const] in {
  def kaddqi : X86Builtin<"unsigned char(unsigned char, unsigned char)">;
  def kaddhi : X86Builtin<"unsigned short(unsigned short, unsigned short)">;
}

let Features = "avx512bw", Attributes = [NoThrow, Const] in {
  def kaddsi : X86Builtin<"unsigned int(unsigned int, unsigned int)">;
  def kadddi : X86Builtin<"unsigned long long int(unsigned long long int, unsigned long long int)">;
}

let Features = "avx512dq", Attributes = [NoThrow, Const] in {
  def kandqi : X86Builtin<"unsigned char(unsigned char, unsigned char)">;
}

let Features = "avx512f", Attributes = [NoThrow, Const] in {
  def kandhi : X86Builtin<"unsigned short(unsigned short, unsigned short)">;
}

let Features = "avx512bw", Attributes = [NoThrow, Const] in {
  def kandsi : X86Builtin<"unsigned int(unsigned int, unsigned int)">;
  def kanddi : X86Builtin<"unsigned long long int(unsigned long long int, unsigned long long int)">;
}

let Features = "avx512dq", Attributes = [NoThrow, Const] in {
  def kandnqi : X86Builtin<"unsigned char(unsigned char, unsigned char)">;
}

let Features = "avx512f", Attributes = [NoThrow, Const] in {
  def kandnhi : X86Builtin<"unsigned short(unsigned short, unsigned short)">;
}

let Features = "avx512bw", Attributes = [NoThrow, Const] in {
  def kandnsi : X86Builtin<"unsigned int(unsigned int, unsigned int)">;
  def kandndi : X86Builtin<"unsigned long long int(unsigned long long int, unsigned long long int)">;
}

let Features = "avx512dq", Attributes = [NoThrow, Const] in {
  def korqi : X86Builtin<"unsigned char(unsigned char, unsigned char)">;
}

let Features = "avx512f", Attributes = [NoThrow, Const] in {
  def korhi : X86Builtin<"unsigned short(unsigned short, unsigned short)">;
}

let Features = "avx512bw", Attributes = [NoThrow, Const] in {
  def korsi : X86Builtin<"unsigned int(unsigned int, unsigned int)">;
  def kordi : X86Builtin<"unsigned long long int(unsigned long long int, unsigned long long int)">;
}

let Features = "avx512dq", Attributes = [NoThrow, Const] in {
  def kortestcqi : X86Builtin<"int(unsigned char, unsigned char)">;
  def kortestzqi : X86Builtin<"int(unsigned char, unsigned char)">;
}

let Features = "avx512f", Attributes = [NoThrow, Const] in {
  def kortestchi : X86Builtin<"int(unsigned short, unsigned short)">;
  def kortestzhi : X86Builtin<"int(unsigned short, unsigned short)">;
}

let Features = "avx512bw", Attributes = [NoThrow, Const] in {
  def kortestcsi : X86Builtin<"int(unsigned int, unsigned int)">;
  def kortestzsi : X86Builtin<"int(unsigned int, unsigned int)">;
  def kortestcdi : X86Builtin<"int(unsigned long long int, unsigned long long int)">;
  def kortestzdi : X86Builtin<"int(unsigned long long int, unsigned long long int)">;
}

let Features = "avx512dq", Attributes = [NoThrow, Const] in {
  def ktestcqi : X86Builtin<"int(unsigned char, unsigned char)">;
  def ktestzqi : X86Builtin<"int(unsigned char, unsigned char)">;
  def ktestchi : X86Builtin<"int(unsigned short, unsigned short)">;
  def ktestzhi : X86Builtin<"int(unsigned short, unsigned short)">;
}

let Features = "avx512bw", Attributes = [NoThrow, Const] in {
  def ktestcsi : X86Builtin<"int(unsigned int, unsigned int)">;
  def ktestzsi : X86Builtin<"int(unsigned int, unsigned int)">;
  def ktestcdi : X86Builtin<"int(unsigned long long int, unsigned long long int)">;
  def ktestzdi : X86Builtin<"int(unsigned long long int, unsigned long long int)">;
}

let Features = "avx512f", Attributes = [NoThrow, Const] in {
  def kunpckhi : X86Builtin<"unsigned short(unsigned short, unsigned short)">;
}

let Features = "avx512dq", Attributes = [NoThrow, Const] in {
  def kxnorqi : X86Builtin<"unsigned char(unsigned char, unsigned char)">;
}

let Features = "avx512f", Attributes = [NoThrow, Const] in {
  def kxnorhi : X86Builtin<"unsigned short(unsigned short, unsigned short)">;
}

let Features = "avx512bw", Attributes = [NoThrow, Const] in {
  def kxnorsi : X86Builtin<"unsigned int(unsigned int, unsigned int)">;
  def kxnordi : X86Builtin<"unsigned long long int(unsigned long long int, unsigned long long int)">;
}

let Features = "avx512dq", Attributes = [NoThrow, Const] in {
  def kxorqi : X86Builtin<"unsigned char(unsigned char, unsigned char)">;
}

let Features = "avx512f", Attributes = [NoThrow, Const] in {
  def kxorhi : X86Builtin<"unsigned short(unsigned short, unsigned short)">;
}

let Features = "avx512bw", Attributes = [NoThrow, Const] in {
  def kxorsi : X86Builtin<"unsigned int(unsigned int, unsigned int)">;
  def kxordi : X86Builtin<"unsigned long long int(unsigned long long int, unsigned long long int)">;
}

let Features = "avx512dq", Attributes = [NoThrow, Const] in {
  def kshiftliqi : X86Builtin<"unsigned char(unsigned char, _Constant unsigned int)">;
}

let Features = "avx512f", Attributes = [NoThrow, Const] in {
  def kshiftlihi : X86Builtin<"unsigned short(unsigned short, _Constant unsigned int)">;
}

let Features = "avx512bw", Attributes = [NoThrow, Const] in {
  def kshiftlisi : X86Builtin<"unsigned int(unsigned int, _Constant unsigned int)">;
  def kshiftlidi : X86Builtin<"unsigned long long int(unsigned long long int, _Constant unsigned int)">;
}

let Features = "avx512dq", Attributes = [NoThrow, Const] in {
  def kshiftriqi : X86Builtin<"unsigned char(unsigned char, _Constant unsigned int)">;
}

let Features = "avx512f", Attributes = [NoThrow, Const] in {
  def kshiftrihi : X86Builtin<"unsigned short(unsigned short, _Constant unsigned int)">;
}

let Features = "avx512bw", Attributes = [NoThrow, Const] in {
  def kshiftrisi : X86Builtin<"unsigned int(unsigned int, _Constant unsigned int)">;
  def kshiftridi : X86Builtin<"unsigned long long int(unsigned long long int, _Constant unsigned int)">;
}

let Features = "avx512dq", Attributes = [NoThrow, Const] in {
  def kmovb : X86Builtin<"unsigned char(unsigned char)">;
}

let Features = "avx512f", Attributes = [NoThrow, Const] in {
  def kmovw : X86Builtin<"unsigned short(unsigned short)">;
}

let Features = "avx512bw", Attributes = [NoThrow, Const] in {
  def kmovd : X86Builtin<"unsigned int(unsigned int)">;
  def kmovq : X86Builtin<"unsigned long long int(unsigned long long int)">;
}

let Features = "avx512bw,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def palignr512 : X86Builtin<"_Vector<64, char>(_Vector<64, char>, _Vector<64, char>, _Constant int)">;
}

let Features = "avx512bw,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def dbpsadbw128 : X86Builtin<"_Vector<8, short>(_Vector<16, char>, _Vector<16, char>, _Constant int)">;
}

let Features = "avx512bw,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def dbpsadbw256 : X86Builtin<"_Vector<16, short>(_Vector<32, char>, _Vector<32, char>, _Constant int)">;
}

let Features = "avx512bw,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def dbpsadbw512 : X86Builtin<"_Vector<32, short>(_Vector<64, char>, _Vector<64, char>, _Constant int)">;
  def psadbw512 : X86Builtin<"_Vector<8, long long int>(_Vector<64, char>, _Vector<64, char>)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def compressdf512_mask : X86Builtin<"_Vector<8, double>(_Vector<8, double>, _Vector<8, double>, unsigned char)">;
  def compressdi512_mask : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int>, _Vector<8, long long int>, unsigned char)">;
}

let Features = "avx512vbmi2,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def compresshi512_mask : X86Builtin<"_Vector<32, short>(_Vector<32, short>, _Vector<32, short>, unsigned int)">;
  def compressqi512_mask : X86Builtin<"_Vector<64, char>(_Vector<64, char>, _Vector<64, char>, unsigned long long int)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def compresssf512_mask : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Vector<16, float>, unsigned short)">;
  def compresssi512_mask : X86Builtin<"_Vector<16, int>(_Vector<16, int>, _Vector<16, int>, unsigned short)">;
}

let Features = "avx512f", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def cmpsd_mask : X86Builtin<"unsigned char(_Vector<2, double>, _Vector<2, double>, _Constant int, unsigned char, _Constant int)">;
  def cmpss_mask : X86Builtin<"unsigned char(_Vector<4, float>, _Vector<4, float>, _Constant int, unsigned char, _Constant int)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def pshufd512 : X86Builtin<"_Vector<16, int>(_Vector<16, int>, _Constant int)">;
  def expanddf512_mask : X86Builtin<"_Vector<8, double>(_Vector<8, double>, _Vector<8, double>, unsigned char)">;
  def expanddi512_mask : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int>, _Vector<8, long long int>, unsigned char)">;
}

let Features = "avx512vbmi2,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def expandhi512_mask : X86Builtin<"_Vector<32, short>(_Vector<32, short>, _Vector<32, short>, unsigned int)">;
  def expandqi512_mask : X86Builtin<"_Vector<64, char>(_Vector<64, char>, _Vector<64, char>, unsigned long long int)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def expandloaddf512_mask : X86Builtin<"_Vector<8, double>(_Vector<8, double const *>, _Vector<8, double>, unsigned char)">;
  def expandloaddi512_mask : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int const *>, _Vector<8, long long int>, unsigned char)">;
}

let Features = "avx512vbmi2,evex512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def expandloadhi512_mask : X86Builtin<"_Vector<32, short>(_Vector<32, short const *>, _Vector<32, short>, unsigned int)">;
  def expandloadqi512_mask : X86Builtin<"_Vector<64, char>(_Vector<64, char const *>, _Vector<64, char>, unsigned long long int)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def expandloadsf512_mask : X86Builtin<"_Vector<16, float>(_Vector<16, float const *>, _Vector<16, float>, unsigned short)">;
  def expandloadsi512_mask : X86Builtin<"_Vector<16, int>(_Vector<16, int const *>, _Vector<16, int>, unsigned short)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def expandsf512_mask : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Vector<16, float>, unsigned short)">;
  def expandsi512_mask : X86Builtin<"_Vector<16, int>(_Vector<16, int>, _Vector<16, int>, unsigned short)">;
  def cvtps2pd512_mask : X86Builtin<"_Vector<8, double>(_Vector<8, float>, _Vector<8, double>, unsigned char, _Constant int)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def compressstoredf512_mask : X86Builtin<"void(_Vector<8, double *>, _Vector<8, double>, unsigned char)">;
  def compressstoredi512_mask : X86Builtin<"void(_Vector<8, long long int *>, _Vector<8, long long int>, unsigned char)">;
}

let Features = "avx512vbmi2,evex512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def compressstorehi512_mask : X86Builtin<"void(_Vector<32, short *>, _Vector<32, short>, unsigned int)">;
  def compressstoreqi512_mask : X86Builtin<"void(_Vector<64, char *>, _Vector<64, char>, unsigned long long int)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def compressstoresf512_mask : X86Builtin<"void(_Vector<16, float *>, _Vector<16, float>, unsigned short)">;
  def compressstoresi512_mask : X86Builtin<"void(_Vector<16, int *>, _Vector<16, int>, unsigned short)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vcvtph2ps_mask : X86Builtin<"_Vector<4, float>(_Vector<8, short>, _Vector<4, float>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vcvtph2ps256_mask : X86Builtin<"_Vector<8, float>(_Vector<8, short>, _Vector<8, float>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vcvtps2ph_mask : X86Builtin<"_Vector<8, short>(_Vector<4, float>, _Constant int, _Vector<8, short>, unsigned char)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vcvtps2ph256_mask : X86Builtin<"_Vector<8, short>(_Vector<8, float>, _Constant int, _Vector<8, short>, unsigned char)">;
}

let Features = "avx512bw,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def cvtw2mask512 : X86Builtin<"unsigned int(_Vector<32, short>)">;
}

let Features = "avx512bw,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def cvtw2mask128 : X86Builtin<"unsigned char(_Vector<8, short>)">;
}

let Features = "avx512bw,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def cvtw2mask256 : X86Builtin<"unsigned short(_Vector<16, short>)">;
}

let Features = "avx512f", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def cvtsd2ss_round_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<2, double>, _Vector<4, float>, unsigned char, _Constant int)">;
  def cvtsi2ss32 : X86Builtin<"_Vector<4, float>(_Vector<4, float>, int, _Constant int)">;
  def cvtss2sd_round_mask : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<4, float>, _Vector<2, double>, unsigned char, _Constant int)">;
  def cvtusi2ss32 : X86Builtin<"_Vector<4, float>(_Vector<4, float>, unsigned int, _Constant int)">;
}

let Features = "avx512vbmi,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vpmultishiftqb512 : X86Builtin<"_Vector<64, char>(_Vector<64, char>, _Vector<64, char>)">;
}

let Features = "avx512vbmi,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vpmultishiftqb128 : X86Builtin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>)">;
}

let Features = "avx512vbmi,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vpmultishiftqb256 : X86Builtin<"_Vector<32, char>(_Vector<32, char>, _Vector<32, char>)">;
}

let Features = "avx512bf16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def cvtne2ps2bf16_128 : X86Builtin<"_Vector<8, __bf16>(_Vector<4, float>, _Vector<4, float>)">;
}

let Features = "avx512bf16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def cvtne2ps2bf16_256 : X86Builtin<"_Vector<16, __bf16>(_Vector<8, float>, _Vector<8, float>)">;
}

let Features = "avx512bf16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def cvtne2ps2bf16_512 : X86Builtin<"_Vector<32, __bf16>(_Vector<16, float>, _Vector<16, float>)">;
}

let Features = "avx512bf16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def cvtneps2bf16_128_mask : X86Builtin<"_Vector<8, __bf16>(_Vector<4, float>, _Vector<8, __bf16>, unsigned char)">;
}

let Features = "avx512bf16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def cvtneps2bf16_256_mask : X86Builtin<"_Vector<8, __bf16>(_Vector<8, float>, _Vector<8, __bf16>, unsigned char)">;
}

let Features = "avx512bf16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def cvtneps2bf16_512_mask : X86Builtin<"_Vector<16, __bf16>(_Vector<16, float>, _Vector<16, __bf16>, unsigned short)">;
}

let Features = "avx512bf16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def dpbf16ps_128 : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<8, __bf16>, _Vector<8, __bf16>)">;
}

let Features = "avx512bf16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def dpbf16ps_256 : X86Builtin<"_Vector<8, float>(_Vector<8, float>, _Vector<16, __bf16>, _Vector<16, __bf16>)">;
}

let Features = "avx512bf16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def dpbf16ps_512 : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Vector<32, __bf16>, _Vector<32, __bf16>)">;
}

let Features = "avx512bf16", Attributes = [NoThrow, Const] in {
  def cvtsbf162ss_32 : X86Builtin<"float(__bf16)">;
}

let Features = "avx512vp2intersect,evex512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vp2intersect_q_512 : X86Builtin<"void(_Vector<8, long long int>, _Vector<8, long long int>, unsigned char *, unsigned char *)">;
}

let Features = "avx512vp2intersect,avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vp2intersect_q_256 : X86Builtin<"void(_Vector<4, long long int>, _Vector<4, long long int>, unsigned char *, unsigned char *)">;
}

let Features = "avx512vp2intersect,avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vp2intersect_q_128 : X86Builtin<"void(_Vector<2, long long int>, _Vector<2, long long int>, unsigned char *, unsigned char *)">;
}

let Features = "avx512vp2intersect,evex512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vp2intersect_d_512 : X86Builtin<"void(_Vector<16, int>, _Vector<16, int>, unsigned short *, unsigned short *)">;
}

let Features = "avx512vp2intersect,avx512vl", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vp2intersect_d_256 : X86Builtin<"void(_Vector<8, int>, _Vector<8, int>, unsigned char *, unsigned char *)">;
}

let Features = "avx512vp2intersect,avx512vl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vp2intersect_d_128 : X86Builtin<"void(_Vector<4, int>, _Vector<4, int>, unsigned char *, unsigned char *)">;
}

let Features = "avx512fp16", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vcomish : X86Builtin<"int(_Vector<8, _Float16>, _Vector<8, _Float16>, _Constant int, _Constant int)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def addph512 : X86Builtin<"_Vector<32, _Float16>(_Vector<32, _Float16>, _Vector<32, _Float16>, _Constant int)">;
  def subph512 : X86Builtin<"_Vector<32, _Float16>(_Vector<32, _Float16>, _Vector<32, _Float16>, _Constant int)">;
  def mulph512 : X86Builtin<"_Vector<32, _Float16>(_Vector<32, _Float16>, _Vector<32, _Float16>, _Constant int)">;
  def divph512 : X86Builtin<"_Vector<32, _Float16>(_Vector<32, _Float16>, _Vector<32, _Float16>, _Constant int)">;
  def maxph512 : X86Builtin<"_Vector<32, _Float16>(_Vector<32, _Float16>, _Vector<32, _Float16>, _Constant int)">;
  def minph512 : X86Builtin<"_Vector<32, _Float16>(_Vector<32, _Float16>, _Vector<32, _Float16>, _Constant int)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def minph256 : X86Builtin<"_Vector<16, _Float16>(_Vector<16, _Float16>, _Vector<16, _Float16>)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def minph128 : X86Builtin<"_Vector<8, _Float16>(_Vector<8, _Float16>, _Vector<8, _Float16>)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def maxph256 : X86Builtin<"_Vector<16, _Float16>(_Vector<16, _Float16>, _Vector<16, _Float16>)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def maxph128 : X86Builtin<"_Vector<8, _Float16>(_Vector<8, _Float16>, _Vector<8, _Float16>)">;
}

let Features = "avx512fp16", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def addsh_round_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<8, _Float16>, _Vector<8, _Float16>, _Vector<8, _Float16>, unsigned char, _Constant int)">;
  def divsh_round_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<8, _Float16>, _Vector<8, _Float16>, _Vector<8, _Float16>, unsigned char, _Constant int)">;
  def mulsh_round_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<8, _Float16>, _Vector<8, _Float16>, _Vector<8, _Float16>, unsigned char, _Constant int)">;
  def subsh_round_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<8, _Float16>, _Vector<8, _Float16>, _Vector<8, _Float16>, unsigned char, _Constant int)">;
  def maxsh_round_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<8, _Float16>, _Vector<8, _Float16>, _Vector<8, _Float16>, unsigned char, _Constant int)">;
  def minsh_round_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<8, _Float16>, _Vector<8, _Float16>, _Vector<8, _Float16>, unsigned char, _Constant int)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def cmpph512_mask : X86Builtin<"unsigned int(_Vector<32, _Float16>, _Vector<32, _Float16>, _Constant int, unsigned int, _Constant int)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def cmpph256_mask : X86Builtin<"unsigned short(_Vector<16, _Float16>, _Vector<16, _Float16>, _Constant int, unsigned short)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def cmpph128_mask : X86Builtin<"unsigned char(_Vector<8, _Float16>, _Vector<8, _Float16>, _Constant int, unsigned char)">;
}

let Features = "avx512fp16", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def cmpsh_mask : X86Builtin<"unsigned char(_Vector<8, _Float16>, _Vector<8, _Float16>, _Constant int, unsigned char, _Constant int)">;
}

let Features = "avx512fp16", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def loadsh128_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<8, _Float16 const *>, _Vector<8, _Float16>, unsigned char)">;
  def storesh128_mask : X86Builtin<"void(_Vector<8, _Float16 *>, _Vector<8, _Float16>, unsigned char)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def rcpph128_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<8, _Float16>, _Vector<8, _Float16>, unsigned char)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def rcpph256_mask : X86Builtin<"_Vector<16, _Float16>(_Vector<16, _Float16>, _Vector<16, _Float16>, unsigned short)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def rcpph512_mask : X86Builtin<"_Vector<32, _Float16>(_Vector<32, _Float16>, _Vector<32, _Float16>, unsigned int)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def rsqrtph128_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<8, _Float16>, _Vector<8, _Float16>, unsigned char)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def rsqrtph256_mask : X86Builtin<"_Vector<16, _Float16>(_Vector<16, _Float16>, _Vector<16, _Float16>, unsigned short)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def rsqrtph512_mask : X86Builtin<"_Vector<32, _Float16>(_Vector<32, _Float16>, _Vector<32, _Float16>, unsigned int)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def getmantph128_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<8, _Float16>, _Constant int, _Vector<8, _Float16>, unsigned char)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def getmantph256_mask : X86Builtin<"_Vector<16, _Float16>(_Vector<16, _Float16>, _Constant int, _Vector<16, _Float16>, unsigned short)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def getmantph512_mask : X86Builtin<"_Vector<32, _Float16>(_Vector<32, _Float16>, _Constant int, _Vector<32, _Float16>, unsigned int, _Constant int)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def getexpph128_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<8, _Float16>, _Vector<8, _Float16>, unsigned char)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def getexpph256_mask : X86Builtin<"_Vector<16, _Float16>(_Vector<16, _Float16>, _Vector<16, _Float16>, unsigned short)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def getexpph512_mask : X86Builtin<"_Vector<32, _Float16>(_Vector<32, _Float16>, _Vector<32, _Float16>, unsigned int, _Constant int)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def scalefph128_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<8, _Float16>, _Vector<8, _Float16>, _Vector<8, _Float16>, unsigned char)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def scalefph256_mask : X86Builtin<"_Vector<16, _Float16>(_Vector<16, _Float16>, _Vector<16, _Float16>, _Vector<16, _Float16>, unsigned short)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def scalefph512_mask : X86Builtin<"_Vector<32, _Float16>(_Vector<32, _Float16>, _Vector<32, _Float16>, _Vector<32, _Float16>, unsigned int, _Constant int)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def rndscaleph_128_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<8, _Float16>, _Constant int, _Vector<8, _Float16>, unsigned char)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def rndscaleph_256_mask : X86Builtin<"_Vector<16, _Float16>(_Vector<16, _Float16>, _Constant int, _Vector<16, _Float16>, unsigned short)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def rndscaleph_mask : X86Builtin<"_Vector<32, _Float16>(_Vector<32, _Float16>, _Constant int, _Vector<32, _Float16>, unsigned int, _Constant int)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def reduceph128_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<8, _Float16>, _Constant int, _Vector<8, _Float16>, unsigned char)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def reduceph256_mask : X86Builtin<"_Vector<16, _Float16>(_Vector<16, _Float16>, _Constant int, _Vector<16, _Float16>, unsigned short)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def reduceph512_mask : X86Builtin<"_Vector<32, _Float16>(_Vector<32, _Float16>, _Constant int, _Vector<32, _Float16>, unsigned int, _Constant int)">;
}

let Features = "avx512fp16", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def rcpsh_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<8, _Float16>, _Vector<8, _Float16>, _Vector<8, _Float16>, unsigned char)">;
  def rsqrtsh_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<8, _Float16>, _Vector<8, _Float16>, _Vector<8, _Float16>, unsigned char)">;
  def getmantsh_round_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<8, _Float16>, _Vector<8, _Float16>, _Constant int, _Vector<8, _Float16>, unsigned char, _Constant int)">;
  def getexpsh128_round_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<8, _Float16>, _Vector<8, _Float16>, _Vector<8, _Float16>, unsigned char, _Constant int)">;
  def scalefsh_round_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<8, _Float16>, _Vector<8, _Float16>, _Vector<8, _Float16>, unsigned char, _Constant int)">;
  def rndscalesh_round_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<8, _Float16>, _Vector<8, _Float16>, _Vector<8, _Float16>, unsigned char, _Constant int, _Constant int)">;
  def reducesh_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<8, _Float16>, _Vector<8, _Float16>, _Vector<8, _Float16>, unsigned char, _Constant int, _Constant int)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def sqrtph : X86Builtin<"_Vector<8, _Float16>(_Vector<8, _Float16>)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def sqrtph256 : X86Builtin<"_Vector<16, _Float16>(_Vector<16, _Float16>)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def sqrtph512 : X86Builtin<"_Vector<32, _Float16>(_Vector<32, _Float16>, _Constant int)">;
}

let Features = "avx512fp16", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def sqrtsh_round_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<8, _Float16>, _Vector<8, _Float16>, _Vector<8, _Float16>, unsigned char, _Constant int)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def fpclassph128_mask : X86Builtin<"unsigned char(_Vector<8, _Float16>, _Constant int, unsigned char)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def fpclassph256_mask : X86Builtin<"unsigned short(_Vector<16, _Float16>, _Constant int, unsigned short)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def fpclassph512_mask : X86Builtin<"unsigned int(_Vector<32, _Float16>, _Constant int, unsigned int)">;
}

let Features = "avx512fp16", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def fpclasssh_mask : X86Builtin<"unsigned char(_Vector<8, _Float16>, _Constant int, unsigned char)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vcvtpd2ph128_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<2, double>, _Vector<8, _Float16>, unsigned char)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vcvtpd2ph256_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<4, double>, _Vector<8, _Float16>, unsigned char)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vcvtpd2ph512_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<8, double>, _Vector<8, _Float16>, unsigned char, _Constant int)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vcvtph2pd128_mask : X86Builtin<"_Vector<2, double>(_Vector<8, _Float16>, _Vector<2, double>, unsigned char)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vcvtph2pd256_mask : X86Builtin<"_Vector<4, double>(_Vector<8, _Float16>, _Vector<4, double>, unsigned char)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vcvtph2pd512_mask : X86Builtin<"_Vector<8, double>(_Vector<8, _Float16>, _Vector<8, double>, unsigned char, _Constant int)">;
}

let Features = "avx512fp16", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vcvtsh2ss_round_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<8, _Float16>, _Vector<4, float>, unsigned char, _Constant int)">;
  def vcvtss2sh_round_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<8, _Float16>, _Vector<4, float>, _Vector<8, _Float16>, unsigned char, _Constant int)">;
  def vcvtsd2sh_round_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<8, _Float16>, _Vector<2, double>, _Vector<8, _Float16>, unsigned char, _Constant int)">;
  def vcvtsh2sd_round_mask : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<8, _Float16>, _Vector<2, double>, unsigned char, _Constant int)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vcvtph2w128_mask : X86Builtin<"_Vector<8, short>(_Vector<8, _Float16>, _Vector<8, short>, unsigned char)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vcvtph2w256_mask : X86Builtin<"_Vector<16, short>(_Vector<16, _Float16>, _Vector<16, short>, unsigned short)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vcvtph2w512_mask : X86Builtin<"_Vector<32, short>(_Vector<32, _Float16>, _Vector<32, short>, unsigned int, _Constant int)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vcvttph2w128_mask : X86Builtin<"_Vector<8, short>(_Vector<8, _Float16>, _Vector<8, short>, unsigned char)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vcvttph2w256_mask : X86Builtin<"_Vector<16, short>(_Vector<16, _Float16>, _Vector<16, short>, unsigned short)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vcvttph2w512_mask : X86Builtin<"_Vector<32, short>(_Vector<32, _Float16>, _Vector<32, short>, unsigned int, _Constant int)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vcvtw2ph128_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<8, short>, _Vector<8, _Float16>, unsigned char)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vcvtw2ph256_mask : X86Builtin<"_Vector<16, _Float16>(_Vector<16, short>, _Vector<16, _Float16>, unsigned short)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vcvtw2ph512_mask : X86Builtin<"_Vector<32, _Float16>(_Vector<32, short>, _Vector<32, _Float16>, unsigned int, _Constant int)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vcvtph2uw128_mask : X86Builtin<"_Vector<8, unsigned short>(_Vector<8, _Float16>, _Vector<8, unsigned short>, unsigned char)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vcvtph2uw256_mask : X86Builtin<"_Vector<16, unsigned short>(_Vector<16, _Float16>, _Vector<16, unsigned short>, unsigned short)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vcvtph2uw512_mask : X86Builtin<"_Vector<32, unsigned short>(_Vector<32, _Float16>, _Vector<32, unsigned short>, unsigned int, _Constant int)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vcvttph2uw128_mask : X86Builtin<"_Vector<8, unsigned short>(_Vector<8, _Float16>, _Vector<8, unsigned short>, unsigned char)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vcvttph2uw256_mask : X86Builtin<"_Vector<16, unsigned short>(_Vector<16, _Float16>, _Vector<16, unsigned short>, unsigned short)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vcvttph2uw512_mask : X86Builtin<"_Vector<32, unsigned short>(_Vector<32, _Float16>, _Vector<32, unsigned short>, unsigned int, _Constant int)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vcvtuw2ph128_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<8, unsigned short>, _Vector<8, _Float16>, unsigned char)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vcvtuw2ph256_mask : X86Builtin<"_Vector<16, _Float16>(_Vector<16, unsigned short>, _Vector<16, _Float16>, unsigned short)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vcvtuw2ph512_mask : X86Builtin<"_Vector<32, _Float16>(_Vector<32, unsigned short>, _Vector<32, _Float16>, unsigned int, _Constant int)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vcvtph2dq128_mask : X86Builtin<"_Vector<4, int>(_Vector<8, _Float16>, _Vector<4, int>, unsigned char)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vcvtph2dq256_mask : X86Builtin<"_Vector<8, int>(_Vector<8, _Float16>, _Vector<8, int>, unsigned char)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vcvtph2dq512_mask : X86Builtin<"_Vector<16, int>(_Vector<16, _Float16>, _Vector<16, int>, unsigned short, _Constant int)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vcvtph2udq128_mask : X86Builtin<"_Vector<4, unsigned int>(_Vector<8, _Float16>, _Vector<4, unsigned int>, unsigned char)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vcvtph2udq256_mask : X86Builtin<"_Vector<8, unsigned int>(_Vector<8, _Float16>, _Vector<8, unsigned int>, unsigned char)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vcvtph2udq512_mask : X86Builtin<"_Vector<16, unsigned int>(_Vector<16, _Float16>, _Vector<16, unsigned int>, unsigned short, _Constant int)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vcvtdq2ph128_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<4, int>, _Vector<8, _Float16>, unsigned char)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vcvtdq2ph256_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<8, int>, _Vector<8, _Float16>, unsigned char)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vcvtdq2ph512_mask : X86Builtin<"_Vector<16, _Float16>(_Vector<16, int>, _Vector<16, _Float16>, unsigned short, _Constant int)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vcvtudq2ph128_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<4, unsigned int>, _Vector<8, _Float16>, unsigned char)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vcvtudq2ph256_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<8, unsigned int>, _Vector<8, _Float16>, unsigned char)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vcvtudq2ph512_mask : X86Builtin<"_Vector<16, _Float16>(_Vector<16, unsigned int>, _Vector<16, _Float16>, unsigned short, _Constant int)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vcvttph2dq128_mask : X86Builtin<"_Vector<4, int>(_Vector<8, _Float16>, _Vector<4, int>, unsigned char)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vcvttph2dq256_mask : X86Builtin<"_Vector<8, int>(_Vector<8, _Float16>, _Vector<8, int>, unsigned char)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vcvttph2dq512_mask : X86Builtin<"_Vector<16, int>(_Vector<16, _Float16>, _Vector<16, int>, unsigned short, _Constant int)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vcvttph2udq128_mask : X86Builtin<"_Vector<4, unsigned int>(_Vector<8, _Float16>, _Vector<4, unsigned int>, unsigned char)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vcvttph2udq256_mask : X86Builtin<"_Vector<8, unsigned int>(_Vector<8, _Float16>, _Vector<8, unsigned int>, unsigned char)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vcvttph2udq512_mask : X86Builtin<"_Vector<16, unsigned int>(_Vector<16, _Float16>, _Vector<16, unsigned int>, unsigned short, _Constant int)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vcvtqq2ph128_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<2, long long int>, _Vector<8, _Float16>, unsigned char)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vcvtqq2ph256_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<4, long long int>, _Vector<8, _Float16>, unsigned char)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vcvtqq2ph512_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<8, long long int>, _Vector<8, _Float16>, unsigned char, _Constant int)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vcvtph2qq128_mask : X86Builtin<"_Vector<2, long long int>(_Vector<8, _Float16>, _Vector<2, long long int>, unsigned char)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vcvtph2qq256_mask : X86Builtin<"_Vector<4, long long int>(_Vector<8, _Float16>, _Vector<4, long long int>, unsigned char)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vcvtph2qq512_mask : X86Builtin<"_Vector<8, long long int>(_Vector<8, _Float16>, _Vector<8, long long int>, unsigned char, _Constant int)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vcvtuqq2ph128_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<2, unsigned long long int>, _Vector<8, _Float16>, unsigned char)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vcvtuqq2ph256_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<4, unsigned long long int>, _Vector<8, _Float16>, unsigned char)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vcvtuqq2ph512_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<8, unsigned long long int>, _Vector<8, _Float16>, unsigned char, _Constant int)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vcvtph2uqq128_mask : X86Builtin<"_Vector<2, unsigned long long int>(_Vector<8, _Float16>, _Vector<2, unsigned long long int>, unsigned char)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vcvtph2uqq256_mask : X86Builtin<"_Vector<4, unsigned long long int>(_Vector<8, _Float16>, _Vector<4, unsigned long long int>, unsigned char)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vcvtph2uqq512_mask : X86Builtin<"_Vector<8, unsigned long long int>(_Vector<8, _Float16>, _Vector<8, unsigned long long int>, unsigned char, _Constant int)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vcvttph2qq128_mask : X86Builtin<"_Vector<2, long long int>(_Vector<8, _Float16>, _Vector<2, long long int>, unsigned char)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vcvttph2qq256_mask : X86Builtin<"_Vector<4, long long int>(_Vector<8, _Float16>, _Vector<4, long long int>, unsigned char)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vcvttph2qq512_mask : X86Builtin<"_Vector<8, long long int>(_Vector<8, _Float16>, _Vector<8, long long int>, unsigned char, _Constant int)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vcvttph2uqq128_mask : X86Builtin<"_Vector<2, unsigned long long int>(_Vector<8, _Float16>, _Vector<2, unsigned long long int>, unsigned char)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vcvttph2uqq256_mask : X86Builtin<"_Vector<4, unsigned long long int>(_Vector<8, _Float16>, _Vector<4, unsigned long long int>, unsigned char)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vcvttph2uqq512_mask : X86Builtin<"_Vector<8, unsigned long long int>(_Vector<8, _Float16>, _Vector<8, unsigned long long int>, unsigned char, _Constant int)">;
}

let Features = "avx512fp16", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vcvtsh2si32 : X86Builtin<"int(_Vector<8, _Float16>, _Constant int)">;
  def vcvtsh2usi32 : X86Builtin<"unsigned int(_Vector<8, _Float16>, _Constant int)">;
  def vcvtusi2sh : X86Builtin<"_Vector<8, _Float16>(_Vector<8, _Float16>, unsigned int, _Constant int)">;
  def vcvtsi2sh : X86Builtin<"_Vector<8, _Float16>(_Vector<8, _Float16>, int, _Constant int)">;
  def vcvttsh2si32 : X86Builtin<"int(_Vector<8, _Float16>, _Constant int)">;
  def vcvttsh2usi32 : X86Builtin<"unsigned int(_Vector<8, _Float16>, _Constant int)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vcvtph2psx128_mask : X86Builtin<"_Vector<4, float>(_Vector<8, _Float16>, _Vector<4, float>, unsigned char)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vcvtph2psx256_mask : X86Builtin<"_Vector<8, float>(_Vector<8, _Float16>, _Vector<8, float>, unsigned char)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vcvtph2psx512_mask : X86Builtin<"_Vector<16, float>(_Vector<16, _Float16>, _Vector<16, float>, unsigned short, _Constant int)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vcvtps2phx128_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<4, float>, _Vector<8, _Float16>, unsigned char)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vcvtps2phx256_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<8, float>, _Vector<8, _Float16>, unsigned char)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vcvtps2phx512_mask : X86Builtin<"_Vector<16, _Float16>(_Vector<16, float>, _Vector<16, _Float16>, unsigned short, _Constant int)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vfmaddph512_mask : X86Builtin<"_Vector<32, _Float16>(_Vector<32, _Float16>, _Vector<32, _Float16>, _Vector<32, _Float16>, unsigned int, _Constant int)">;
  def vfmaddph512_mask3 : X86Builtin<"_Vector<32, _Float16>(_Vector<32, _Float16>, _Vector<32, _Float16>, _Vector<32, _Float16>, unsigned int, _Constant int)">;
  def vfmaddph512_maskz : X86Builtin<"_Vector<32, _Float16>(_Vector<32, _Float16>, _Vector<32, _Float16>, _Vector<32, _Float16>, unsigned int, _Constant int)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vfmaddsubph : X86Builtin<"_Vector<8, _Float16>(_Vector<8, _Float16>, _Vector<8, _Float16>, _Vector<8, _Float16>)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vfmaddsubph256 : X86Builtin<"_Vector<16, _Float16>(_Vector<16, _Float16>, _Vector<16, _Float16>, _Vector<16, _Float16>)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vfmaddsubph512_mask : X86Builtin<"_Vector<32, _Float16>(_Vector<32, _Float16>, _Vector<32, _Float16>, _Vector<32, _Float16>, unsigned int, _Constant int)">;
  def vfmaddsubph512_maskz : X86Builtin<"_Vector<32, _Float16>(_Vector<32, _Float16>, _Vector<32, _Float16>, _Vector<32, _Float16>, unsigned int, _Constant int)">;
  def vfmaddsubph512_mask3 : X86Builtin<"_Vector<32, _Float16>(_Vector<32, _Float16>, _Vector<32, _Float16>, _Vector<32, _Float16>, unsigned int, _Constant int)">;
  def vfmsubaddph512_mask3 : X86Builtin<"_Vector<32, _Float16>(_Vector<32, _Float16>, _Vector<32, _Float16>, _Vector<32, _Float16>, unsigned int, _Constant int)">;
  def vfmsubph512_mask3 : X86Builtin<"_Vector<32, _Float16>(_Vector<32, _Float16>, _Vector<32, _Float16>, _Vector<32, _Float16>, unsigned int, _Constant int)">;
}

let Features = "avx512fp16", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vfmaddsh3_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<8, _Float16>, _Vector<8, _Float16>, _Vector<8, _Float16>, unsigned char, _Constant int)">;
  def vfmaddsh3_maskz : X86Builtin<"_Vector<8, _Float16>(_Vector<8, _Float16>, _Vector<8, _Float16>, _Vector<8, _Float16>, unsigned char, _Constant int)">;
  def vfmaddsh3_mask3 : X86Builtin<"_Vector<8, _Float16>(_Vector<8, _Float16>, _Vector<8, _Float16>, _Vector<8, _Float16>, unsigned char, _Constant int)">;
  def vfmsubsh3_mask3 : X86Builtin<"_Vector<8, _Float16>(_Vector<8, _Float16>, _Vector<8, _Float16>, _Vector<8, _Float16>, unsigned char, _Constant int)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vfmaddcph128_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>, unsigned char)">;
  def vfmaddcph128_maskz : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>, unsigned char)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vfmaddcph256_mask : X86Builtin<"_Vector<8, float>(_Vector<8, float>, _Vector<8, float>, _Vector<8, float>, unsigned char)">;
  def vfmaddcph256_maskz : X86Builtin<"_Vector<8, float>(_Vector<8, float>, _Vector<8, float>, _Vector<8, float>, unsigned char)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vfmaddcph512_mask : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Vector<16, float>, _Vector<16, float>, unsigned short, _Constant int)">;
  def vfmaddcph512_maskz : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Vector<16, float>, _Vector<16, float>, unsigned short, _Constant int)">;
  def vfmaddcph512_mask3 : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Vector<16, float>, _Vector<16, float>, unsigned short, _Constant int)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vfcmaddcph128_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>, unsigned char)">;
  def vfcmaddcph128_maskz : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>, unsigned char)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vfcmaddcph256_mask : X86Builtin<"_Vector<8, float>(_Vector<8, float>, _Vector<8, float>, _Vector<8, float>, unsigned char)">;
  def vfcmaddcph256_maskz : X86Builtin<"_Vector<8, float>(_Vector<8, float>, _Vector<8, float>, _Vector<8, float>, unsigned char)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vfcmaddcph512_mask : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Vector<16, float>, _Vector<16, float>, unsigned short, _Constant int)">;
  def vfcmaddcph512_maskz : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Vector<16, float>, _Vector<16, float>, unsigned short, _Constant int)">;
  def vfcmaddcph512_mask3 : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Vector<16, float>, _Vector<16, float>, unsigned short, _Constant int)">;
}

let Features = "avx512fp16", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vfmaddcsh_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>, unsigned char, _Constant int)">;
  def vfmaddcsh_maskz : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>, unsigned char, _Constant int)">;
  def vfcmaddcsh_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>, unsigned char, _Constant int)">;
  def vfcmaddcsh_maskz : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>, unsigned char, _Constant int)">;
  def vfmaddcsh_round_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>, unsigned char, _Constant int)">;
  def vfmaddcsh_round_mask3 : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>, unsigned char, _Constant int)">;
  def vfcmaddcsh_round_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>, unsigned char, _Constant int)">;
  def vfcmaddcsh_round_mask3 : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>, unsigned char, _Constant int)">;
  def vfmulcsh_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>, unsigned char, _Constant int)">;
  def vfcmulcsh_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>, unsigned char, _Constant int)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vfmulcph128_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>, unsigned char)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vfmulcph256_mask : X86Builtin<"_Vector<8, float>(_Vector<8, float>, _Vector<8, float>, _Vector<8, float>, unsigned char)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vfmulcph512_mask : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Vector<16, float>, _Vector<16, float>, unsigned short, _Constant int)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vfcmulcph128_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>, unsigned char)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vfcmulcph256_mask : X86Builtin<"_Vector<8, float>(_Vector<8, float>, _Vector<8, float>, _Vector<8, float>, unsigned char)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vfcmulcph512_mask : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Vector<16, float>, _Vector<16, float>, unsigned short, _Constant int)">;
}

let Features = "avx512bw,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def selectb_128 : X86Builtin<"_Vector<16, char>(unsigned short, _Vector<16, char>, _Vector<16, char>)">;
}

let Features = "avx512bw,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def selectb_256 : X86Builtin<"_Vector<32, char>(unsigned int, _Vector<32, char>, _Vector<32, char>)">;
}

let Features = "avx512bw,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def selectb_512 : X86Builtin<"_Vector<64, char>(unsigned long long int, _Vector<64, char>, _Vector<64, char>)">;
}

let Features = "avx512bw,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def selectw_128 : X86Builtin<"_Vector<8, short>(unsigned char, _Vector<8, short>, _Vector<8, short>)">;
}

let Features = "avx512bw,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def selectw_256 : X86Builtin<"_Vector<16, short>(unsigned short, _Vector<16, short>, _Vector<16, short>)">;
}

let Features = "avx512bw,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def selectw_512 : X86Builtin<"_Vector<32, short>(unsigned int, _Vector<32, short>, _Vector<32, short>)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def selectd_128 : X86Builtin<"_Vector<4, int>(unsigned char, _Vector<4, int>, _Vector<4, int>)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def selectd_256 : X86Builtin<"_Vector<8, int>(unsigned char, _Vector<8, int>, _Vector<8, int>)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def selectd_512 : X86Builtin<"_Vector<16, int>(unsigned short, _Vector<16, int>, _Vector<16, int>)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def selectph_128 : X86Builtin<"_Vector<8, _Float16>(unsigned char, _Vector<8, _Float16>, _Vector<8, _Float16>)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def selectph_256 : X86Builtin<"_Vector<16, _Float16>(unsigned short, _Vector<16, _Float16>, _Vector<16, _Float16>)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def selectph_512 : X86Builtin<"_Vector<32, _Float16>(unsigned int, _Vector<32, _Float16>, _Vector<32, _Float16>)">;
}

let Features = "avx512bf16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def selectpbf_128 : X86Builtin<"_Vector<8, __bf16>(unsigned char, _Vector<8, __bf16>, _Vector<8, __bf16>)">;
}

let Features = "avx512bf16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def selectpbf_256 : X86Builtin<"_Vector<16, __bf16>(unsigned short, _Vector<16, __bf16>, _Vector<16, __bf16>)">;
}

let Features = "avx512bf16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def selectpbf_512 : X86Builtin<"_Vector<32, __bf16>(unsigned int, _Vector<32, __bf16>, _Vector<32, __bf16>)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def selectq_128 : X86Builtin<"_Vector<2, long long int>(unsigned char, _Vector<2, long long int>, _Vector<2, long long int>)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def selectq_256 : X86Builtin<"_Vector<4, long long int>(unsigned char, _Vector<4, long long int>, _Vector<4, long long int>)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def selectq_512 : X86Builtin<"_Vector<8, long long int>(unsigned char, _Vector<8, long long int>, _Vector<8, long long int>)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def selectps_128 : X86Builtin<"_Vector<4, float>(unsigned char, _Vector<4, float>, _Vector<4, float>)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def selectps_256 : X86Builtin<"_Vector<8, float>(unsigned char, _Vector<8, float>, _Vector<8, float>)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def selectps_512 : X86Builtin<"_Vector<16, float>(unsigned short, _Vector<16, float>, _Vector<16, float>)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def selectpd_128 : X86Builtin<"_Vector<2, double>(unsigned char, _Vector<2, double>, _Vector<2, double>)">;
}

let Features = "avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def selectpd_256 : X86Builtin<"_Vector<4, double>(unsigned char, _Vector<4, double>, _Vector<4, double>)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def selectpd_512 : X86Builtin<"_Vector<8, double>(unsigned char, _Vector<8, double>, _Vector<8, double>)">;
}

let Features = "avx512fp16", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def selectsh_128 : X86Builtin<"_Vector<8, _Float16>(unsigned char, _Vector<8, _Float16>, _Vector<8, _Float16>)">;
}

let Features = "avx512bf16", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def selectsbf_128 : X86Builtin<"_Vector<8, __bf16>(unsigned char, _Vector<8, __bf16>, _Vector<8, __bf16>)">;
}

let Features = "avx512f", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def selectss_128 : X86Builtin<"_Vector<4, float>(unsigned char, _Vector<4, float>, _Vector<4, float>)">;
  def selectsd_128 : X86Builtin<"_Vector<2, double>(unsigned char, _Vector<2, double>, _Vector<2, double>)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def reduce_fadd_pd512 : X86Builtin<"double(double, _Vector<8, double>)">;
  def reduce_fadd_ps512 : X86Builtin<"float(float, _Vector<16, float>)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def reduce_fadd_ph512 : X86Builtin<"_Float16(_Float16, _Vector<32, _Float16>)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def reduce_fadd_ph256 : X86Builtin<"_Float16(_Float16, _Vector<16, _Float16>)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def reduce_fadd_ph128 : X86Builtin<"_Float16(_Float16, _Vector<8, _Float16>)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def reduce_fmax_pd512 : X86Builtin<"double(_Vector<8, double>)">;
  def reduce_fmax_ps512 : X86Builtin<"float(_Vector<16, float>)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def reduce_fmax_ph512 : X86Builtin<"_Float16(_Vector<32, _Float16>)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def reduce_fmax_ph256 : X86Builtin<"_Float16(_Vector<16, _Float16>)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def reduce_fmax_ph128 : X86Builtin<"_Float16(_Vector<8, _Float16>)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def reduce_fmin_pd512 : X86Builtin<"double(_Vector<8, double>)">;
  def reduce_fmin_ps512 : X86Builtin<"float(_Vector<16, float>)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def reduce_fmin_ph512 : X86Builtin<"_Float16(_Vector<32, _Float16>)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def reduce_fmin_ph256 : X86Builtin<"_Float16(_Vector<16, _Float16>)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def reduce_fmin_ph128 : X86Builtin<"_Float16(_Vector<8, _Float16>)">;
}

let Features = "avx512f,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def reduce_fmul_pd512 : X86Builtin<"double(double, _Vector<8, double>)">;
  def reduce_fmul_ps512 : X86Builtin<"float(float, _Vector<16, float>)">;
}

let Features = "avx512fp16,evex512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def reduce_fmul_ph512 : X86Builtin<"_Float16(_Float16, _Vector<32, _Float16>)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def reduce_fmul_ph256 : X86Builtin<"_Float16(_Float16, _Vector<16, _Float16>)">;
}

let Features = "avx512fp16,avx512vl", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def reduce_fmul_ph128 : X86Builtin<"_Float16(_Float16, _Vector<8, _Float16>)">;
}

let Features = "mwaitx", Attributes = [NoThrow] in {
  def monitorx : X86Builtin<"void(void const *, unsigned int, unsigned int)">;
  def mwaitx : X86Builtin<"void(unsigned int, unsigned int, unsigned int)">;
}

let Features = "waitpkg", Attributes = [NoThrow] in {
  def umonitor : X86Builtin<"void(void const *)">;
  def umwait : X86Builtin<"unsigned char(unsigned int, unsigned int, unsigned int)">;
  def tpause : X86Builtin<"unsigned char(unsigned int, unsigned int, unsigned int)">;
}

let Features = "clzero", Attributes = [NoThrow] in {
  def clzero : X86Builtin<"void(void *)">;
}

let Features = "cldemote", Attributes = [NoThrow] in {
  def cldemote : X86Builtin<"void(void const *)">;
}

let Features = "movdiri", Attributes = [NoThrow] in {
  def directstore_u32 : X86Builtin<"void(unsigned int *, unsigned int)">;
}

let Features = "movdir64b", Attributes = [NoThrow] in {
  def movdir64b : X86Builtin<"void(void *, void const *)">;
}

let Features = "ptwrite", Attributes = [NoThrow] in {
  def ptwrite32 : X86Builtin<"void(unsigned int)">;
}

let Features = "invpcid", Attributes = [NoThrow, Const] in {
  def invpcid : X86Builtin<"void(unsigned int, void *)">;
}

let Features = "enqcmd", Attributes = [NoThrow] in {
  def enqcmd : X86Builtin<"unsigned char(void *, void const *)">;
  def enqcmds : X86Builtin<"unsigned char(void *, void const *)">;
}

let Features = "kl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def loadiwkey : X86Builtin<"void(_Vector<2, long long int>, _Vector<2, long long int>, _Vector<2, long long int>, unsigned int)">;
  def encodekey128_u32 : X86Builtin<"unsigned int(unsigned int, _Vector<2, long long int>, void *)">;
  def encodekey256_u32 : X86Builtin<"unsigned int(unsigned int, _Vector<2, long long int>, _Vector<2, long long int>, void *)">;
  def aesenc128kl_u8 : X86Builtin<"unsigned char(_Vector<2, long long int *>, _Vector<2, long long int>, void const *)">;
  def aesenc256kl_u8 : X86Builtin<"unsigned char(_Vector<2, long long int *>, _Vector<2, long long int>, void const *)">;
  def aesdec128kl_u8 : X86Builtin<"unsigned char(_Vector<2, long long int *>, _Vector<2, long long int>, void const *)">;
  def aesdec256kl_u8 : X86Builtin<"unsigned char(_Vector<2, long long int *>, _Vector<2, long long int>, void const *)">;
}

let Features = "kl,widekl", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def aesencwide128kl_u8 : X86Builtin<"unsigned char(_Vector<2, long long int *>, _Vector<2, long long int const *>, void const *)">;
  def aesencwide256kl_u8 : X86Builtin<"unsigned char(_Vector<2, long long int *>, _Vector<2, long long int const *>, void const *)">;
  def aesdecwide128kl_u8 : X86Builtin<"unsigned char(_Vector<2, long long int *>, _Vector<2, long long int const *>, void const *)">;
  def aesdecwide256kl_u8 : X86Builtin<"unsigned char(_Vector<2, long long int *>, _Vector<2, long long int const *>, void const *)">;
}

let Features = "serialize", Attributes = [NoThrow] in {
  def serialize : X86Builtin<"void()">;
}

let Features = "tsxldtrk", Attributes = [NoThrow] in {
  def xsusldtrk : X86Builtin<"void()">;
  def xresldtrk : X86Builtin<"void()">;
}

let Features = "raoint", Attributes = [NoThrow] in {
  def aadd32 : X86Builtin<"void(void *, signed int)">;
  def aand32 : X86Builtin<"void(void *, signed int)">;
  def aor32 : X86Builtin<"void(void *, signed int)">;
  def axor32 : X86Builtin<"void(void *, signed int)">;
}

let Header = "intrin.h", Languages = "ALL_MS_LANGUAGES", Attributes = [NoThrow, RequireDeclaration] in {
  def _BitScanForward : X86LibBuiltin<"unsigned char(msuint32_t *, msuint32_t)">;
  def _BitScanReverse : X86LibBuiltin<"unsigned char(msuint32_t *, msuint32_t)">;
  def _ReadWriteBarrier : X86LibBuiltin<"void()">;
  def _ReadBarrier : X86LibBuiltin<"void()">;
  def _WriteBarrier : X86LibBuiltin<"void()">;
  def __cpuid : X86LibBuiltin<"void(int *, int)">;
  def __cpuidex : X86LibBuiltin<"void(int *, int, int)">;
}

let Header = "intrin.h", Languages = "ALL_MS_LANGUAGES", Attributes = [NoThrow, Const, RequireDeclaration] in {
  def __emul : X86LibBuiltin<"long long int(int, int)">;
  def __emulu : X86LibBuiltin<"unsigned long long int(unsigned int, unsigned int)">;
}

let Header = "intrin.h", Languages = "ALL_MS_LANGUAGES", Attributes = [NoThrow, RequireDeclaration] in {
  def _AddressOfReturnAddress : X86LibBuiltin<"void *()">;
  def __stosb : X86LibBuiltin<"void(unsigned char *, unsigned char, size_t)">;
}

let Header = "intrin.h", Languages = "ALL_MS_LANGUAGES", Attributes = [NoThrow, RequireDeclaration, NoReturn] in {
  def __int2c : X86LibBuiltin<"void()">;
  def __ud2 : X86LibBuiltin<"void()">;
}

let Header = "intrin.h", Languages = "ALL_MS_LANGUAGES", Attributes = [NoThrow, RequireDeclaration] in {
  def __readfsbyte : X86LibBuiltin<"unsigned char(msuint32_t)">;
  def __readfsword : X86LibBuiltin<"unsigned short(msuint32_t)">;
  def __readfsdword : X86LibBuiltin<"msuint32_t(msuint32_t)">;
  def __readfsqword : X86LibBuiltin<"unsigned long long int(msuint32_t)">;
  def __readgsbyte : X86LibBuiltin<"unsigned char(msuint32_t)">;
  def __readgsword : X86LibBuiltin<"unsigned short(msuint32_t)">;
  def __readgsdword : X86LibBuiltin<"msuint32_t(msuint32_t)">;
  def __readgsqword : X86LibBuiltin<"unsigned long long int(msuint32_t)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vdpphps128 : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<8, _Float16>, _Vector<8, _Float16>)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vdpphps256 : X86Builtin<"_Vector<8, float>(_Vector<8, float>, _Vector<16, _Float16>, _Vector<16, _Float16>)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vdpphps512 : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Vector<32, _Float16>, _Vector<32, _Float16>)">;
  def vpdpbssd512 : X86Builtin<"_Vector<16, int>(_Vector<16, int>, _Vector<16, int>, _Vector<16, int>)">;
  def vpdpbssds512 : X86Builtin<"_Vector<16, int>(_Vector<16, int>, _Vector<16, int>, _Vector<16, int>)">;
  def vpdpbsud512 : X86Builtin<"_Vector<16, int>(_Vector<16, int>, _Vector<16, int>, _Vector<16, int>)">;
  def vpdpbsuds512 : X86Builtin<"_Vector<16, int>(_Vector<16, int>, _Vector<16, int>, _Vector<16, int>)">;
  def vpdpbuud512 : X86Builtin<"_Vector<16, int>(_Vector<16, int>, _Vector<16, int>, _Vector<16, int>)">;
  def vpdpbuuds512 : X86Builtin<"_Vector<16, int>(_Vector<16, int>, _Vector<16, int>, _Vector<16, int>)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vpdpwsud512 : X86Builtin<"_Vector<16, int>(_Vector<16, int>, _Vector<16, int>, _Vector<16, int>)">;
  def vpdpwsuds512 : X86Builtin<"_Vector<16, int>(_Vector<16, int>, _Vector<16, int>, _Vector<16, int>)">;
  def vpdpwusd512 : X86Builtin<"_Vector<16, int>(_Vector<16, int>, _Vector<16, int>, _Vector<16, int>)">;
  def vpdpwusds512 : X86Builtin<"_Vector<16, int>(_Vector<16, int>, _Vector<16, int>, _Vector<16, int>)">;
  def vpdpwuud512 : X86Builtin<"_Vector<16, int>(_Vector<16, int>, _Vector<16, int>, _Vector<16, int>)">;
  def vpdpwuuds512 : X86Builtin<"_Vector<16, int>(_Vector<16, int>, _Vector<16, int>, _Vector<16, int>)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def mpsadbw512 : X86Builtin<"_Vector<32, short>(_Vector<64, char>, _Vector<64, char>, _Constant char)">;
}

let Features = "avxvnniint16|avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vpdpwsud128 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>, _Vector<4, int>)">;
}

let Features = "avxvnniint16|avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vpdpwsud256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<8, int>, _Vector<8, int>)">;
}

let Features = "avxvnniint16|avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vpdpwsuds128 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>, _Vector<4, int>)">;
}

let Features = "avxvnniint16|avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vpdpwsuds256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<8, int>, _Vector<8, int>)">;
}

let Features = "avxvnniint16|avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vpdpwusd128 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>, _Vector<4, int>)">;
}

let Features = "avxvnniint16|avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vpdpwusd256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<8, int>, _Vector<8, int>)">;
}

let Features = "avxvnniint16|avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vpdpwusds128 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>, _Vector<4, int>)">;
}

let Features = "avxvnniint16|avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vpdpwusds256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<8, int>, _Vector<8, int>)">;
}

let Features = "avxvnniint16|avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vpdpwuud128 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>, _Vector<4, int>)">;
}

let Features = "avxvnniint16|avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vpdpwuud256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<8, int>, _Vector<8, int>)">;
}

let Features = "avxvnniint16|avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vpdpwuuds128 : X86Builtin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>, _Vector<4, int>)">;
}

let Features = "avxvnniint16|avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vpdpwuuds256 : X86Builtin<"_Vector<8, int>(_Vector<8, int>, _Vector<8, int>, _Vector<8, int>)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vcvttsd2sis32 : X86Builtin<"int(_Vector<2, double>, _Constant int)">;
  def vcvttsd2usis32 : X86Builtin<"unsigned int(_Vector<2, double>, _Constant int)">;
  def vcvttss2sis32 : X86Builtin<"int(_Vector<4, float>, _Constant int)">;
  def vcvttss2usis32 : X86Builtin<"unsigned int(_Vector<4, float>, _Constant int)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vcvttpd2dqs128_mask : X86Builtin<"_Vector<4, int>(_Vector<2, double>, _Vector<4, int>, unsigned char)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vcvttpd2dqs256_mask : X86Builtin<"_Vector<4, int>(_Vector<4, double>, _Vector<4, int>, unsigned char)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vcvttpd2dqs512_round_mask : X86Builtin<"_Vector<8, int>(_Vector<8, double>, _Vector<8, int>, unsigned char, _Constant int)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vcvttpd2udqs128_mask : X86Builtin<"_Vector<4, int>(_Vector<2, double>, _Vector<4, int>, unsigned char)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vcvttpd2udqs256_mask : X86Builtin<"_Vector<4, int>(_Vector<4, double>, _Vector<4, int>, unsigned char)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vcvttpd2udqs512_round_mask : X86Builtin<"_Vector<8, int>(_Vector<8, double>, _Vector<8, int>, unsigned char, _Constant int)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vcvttpd2qqs128_mask : X86Builtin<"_Vector<2, long long int>(_Vector<2, double>, _Vector<2, long long int>, unsigned char)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vcvttpd2qqs256_mask : X86Builtin<"_Vector<4, long long int>(_Vector<4, double>, _Vector<4, long long int>, unsigned char)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vcvttpd2qqs512_round_mask : X86Builtin<"_Vector<8, long long int>(_Vector<8, double>, _Vector<8, long long int>, unsigned char, _Constant int)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vcvttpd2uqqs128_mask : X86Builtin<"_Vector<2, long long int>(_Vector<2, double>, _Vector<2, long long int>, unsigned char)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vcvttpd2uqqs256_mask : X86Builtin<"_Vector<4, long long int>(_Vector<4, double>, _Vector<4, long long int>, unsigned char)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vcvttpd2uqqs512_round_mask : X86Builtin<"_Vector<8, long long int>(_Vector<8, double>, _Vector<8, long long int>, unsigned char, _Constant int)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vcvttps2dqs128_mask : X86Builtin<"_Vector<4, int>(_Vector<4, float>, _Vector<4, int>, unsigned char)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vcvttps2dqs256_mask : X86Builtin<"_Vector<8, int>(_Vector<8, float>, _Vector<8, int>, unsigned char)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vcvttps2dqs512_round_mask : X86Builtin<"_Vector<16, int>(_Vector<16, float>, _Vector<16, int>, unsigned short, _Constant int)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vcvttps2udqs128_mask : X86Builtin<"_Vector<4, int>(_Vector<4, float>, _Vector<4, int>, unsigned char)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vcvttps2udqs256_mask : X86Builtin<"_Vector<8, int>(_Vector<8, float>, _Vector<8, int>, unsigned char)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vcvttps2udqs512_round_mask : X86Builtin<"_Vector<16, int>(_Vector<16, float>, _Vector<16, int>, unsigned short, _Constant int)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vcvttps2qqs128_mask : X86Builtin<"_Vector<2, long long int>(_Vector<4, float>, _Vector<2, long long int>, unsigned char)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vcvttps2qqs256_mask : X86Builtin<"_Vector<4, long long int>(_Vector<4, float>, _Vector<4, long long int>, unsigned char)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vcvttps2qqs512_round_mask : X86Builtin<"_Vector<8, long long int>(_Vector<8, float>, _Vector<8, long long int>, unsigned char, _Constant int)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vcvttps2uqqs128_mask : X86Builtin<"_Vector<2, long long int>(_Vector<4, float>, _Vector<2, long long int>, unsigned char)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vcvttps2uqqs256_mask : X86Builtin<"_Vector<4, long long int>(_Vector<4, float>, _Vector<4, long long int>, unsigned char)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vcvttps2uqqs512_round_mask : X86Builtin<"_Vector<8, long long int>(_Vector<8, float>, _Vector<8, long long int>, unsigned char, _Constant int)">;
}

let Features = "avxneconvert", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vbcstnebf162ps128 : X86Builtin<"_Vector<4, float>(__bf16 const *)">;
}

let Features = "avxneconvert", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vbcstnebf162ps256 : X86Builtin<"_Vector<8, float>(__bf16 const *)">;
}

let Features = "avxneconvert", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vbcstnesh2ps128 : X86Builtin<"_Vector<4, float>(_Float16 const *)">;
}

let Features = "avxneconvert", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vbcstnesh2ps256 : X86Builtin<"_Vector<8, float>(_Float16 const *)">;
}

let Features = "avxneconvert", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vcvtneebf162ps128 : X86Builtin<"_Vector<4, float>(_Vector<8, __bf16 const *>)">;
}

let Features = "avxneconvert", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vcvtneebf162ps256 : X86Builtin<"_Vector<8, float>(_Vector<16, __bf16 const *>)">;
}

let Features = "avxneconvert", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vcvtneeph2ps128 : X86Builtin<"_Vector<4, float>(_Vector<8, _Float16 const *>)">;
}

let Features = "avxneconvert", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vcvtneeph2ps256 : X86Builtin<"_Vector<8, float>(_Vector<16, _Float16 const *>)">;
}

let Features = "avxneconvert", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vcvtneobf162ps128 : X86Builtin<"_Vector<4, float>(_Vector<8, __bf16 const *>)">;
}

let Features = "avxneconvert", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vcvtneobf162ps256 : X86Builtin<"_Vector<8, float>(_Vector<16, __bf16 const *>)">;
}

let Features = "avxneconvert", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vcvtneoph2ps128 : X86Builtin<"_Vector<4, float>(_Vector<8, _Float16 const *>)">;
}

let Features = "avxneconvert", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vcvtneoph2ps256 : X86Builtin<"_Vector<8, float>(_Vector<16, _Float16 const *>)">;
}

let Features = "avx512bf16,avx512vl|avxneconvert", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vcvtneps2bf16128 : X86Builtin<"_Vector<8, __bf16>(_Vector<4, float>)">;
}

let Features = "avx512bf16,avx512vl|avxneconvert", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vcvtneps2bf16256 : X86Builtin<"_Vector<8, __bf16>(_Vector<8, float>)">;
}

let Features = "sha512", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vsha512msg1 : X86Builtin<"_Vector<4, unsigned long long int>(_Vector<4, unsigned long long int>, _Vector<2, unsigned long long int>)">;
  def vsha512msg2 : X86Builtin<"_Vector<4, unsigned long long int>(_Vector<4, unsigned long long int>, _Vector<4, unsigned long long int>)">;
  def vsha512rnds2 : X86Builtin<"_Vector<4, unsigned long long int>(_Vector<4, unsigned long long int>, _Vector<4, unsigned long long int>, _Vector<2, unsigned long long int>)">;
}

let Header = "intrin.h", Languages = "ALL_MS_LANGUAGES", Attributes = [NoThrow, RequireDeclaration] in {
  def _InterlockedAnd64 : X86LibBuiltin<"int64_t(int64_t volatile *, int64_t)">;
  def _InterlockedDecrement64 : X86LibBuiltin<"int64_t(int64_t volatile *)">;
  def _InterlockedExchange64 : X86LibBuiltin<"int64_t(int64_t volatile *, int64_t)">;
  def _InterlockedExchangeAdd64 : X86LibBuiltin<"int64_t(int64_t volatile *, int64_t)">;
  def _InterlockedExchangeSub64 : X86LibBuiltin<"int64_t(int64_t volatile *, int64_t)">;
  def _InterlockedIncrement64 : X86LibBuiltin<"int64_t(int64_t volatile *)">;
  def _InterlockedOr64 : X86LibBuiltin<"int64_t(int64_t volatile *, int64_t)">;
  def _InterlockedXor64 : X86LibBuiltin<"int64_t(int64_t volatile *, int64_t)">;
}

let Features = "sm3", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vsm3msg1 : X86Builtin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, _Vector<4, unsigned int>)">;
  def vsm3msg2 : X86Builtin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, _Vector<4, unsigned int>)">;
  def vsm3rnds2 : X86Builtin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, _Vector<4, unsigned int>, _Constant unsigned int)">;
}

let Features = "sm4", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vsm4key4128 : X86Builtin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)">;
}

let Features = "sm4", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vsm4key4256 : X86Builtin<"_Vector<8, unsigned int>(_Vector<8, unsigned int>, _Vector<8, unsigned int>)">;
}

let Features = "sm4", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vsm4rnds4128 : X86Builtin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)">;
}

let Features = "sm4", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vsm4rnds4256 : X86Builtin<"_Vector<8, unsigned int>(_Vector<8, unsigned int>, _Vector<8, unsigned int>)">;
}

let Features = "avx10.2-512,sm4", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vsm4key4512 : X86Builtin<"_Vector<16, unsigned int>(_Vector<16, unsigned int>, _Vector<16, unsigned int>)">;
  def vsm4rnds4512 : X86Builtin<"_Vector<16, unsigned int>(_Vector<16, unsigned int>, _Vector<16, unsigned int>)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vminmaxbf16128 : X86Builtin<"_Vector<8, __bf16>(_Vector<8, __bf16>, _Vector<8, __bf16>, _Constant int)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vminmaxbf16256 : X86Builtin<"_Vector<16, __bf16>(_Vector<16, __bf16>, _Vector<16, __bf16>, _Constant int)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vminmaxbf16512 : X86Builtin<"_Vector<32, __bf16>(_Vector<32, __bf16>, _Vector<32, __bf16>, _Constant int)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vminmaxpd128_mask : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Constant int, _Vector<2, double>, unsigned char)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vminmaxpd256_mask : X86Builtin<"_Vector<4, double>(_Vector<4, double>, _Vector<4, double>, _Constant int, _Vector<4, double>, unsigned char)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vminmaxpd512_round_mask : X86Builtin<"_Vector<8, double>(_Vector<8, double>, _Vector<8, double>, _Constant int, _Vector<8, double>, unsigned char, _Constant int)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vminmaxph128_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<8, _Float16>, _Vector<8, _Float16>, _Constant int, _Vector<8, _Float16>, unsigned char)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vminmaxph256_mask : X86Builtin<"_Vector<16, _Float16>(_Vector<16, _Float16>, _Vector<16, _Float16>, _Constant int, _Vector<16, _Float16>, unsigned short)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vminmaxph512_round_mask : X86Builtin<"_Vector<32, _Float16>(_Vector<32, _Float16>, _Vector<32, _Float16>, _Constant int, _Vector<32, _Float16>, unsigned int, _Constant int)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vminmaxps128_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Constant int, _Vector<4, float>, unsigned char)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vminmaxps256_mask : X86Builtin<"_Vector<8, float>(_Vector<8, float>, _Vector<8, float>, _Constant int, _Vector<8, float>, unsigned char)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vminmaxps512_round_mask : X86Builtin<"_Vector<16, float>(_Vector<16, float>, _Vector<16, float>, _Constant int, _Vector<16, float>, unsigned short, _Constant int)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vminmaxsd_round_mask : X86Builtin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Constant int, _Vector<2, double>, unsigned char, _Constant int)">;
  def vminmaxsh_round_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<8, _Float16>, _Vector<8, _Float16>, _Constant int, _Vector<8, _Float16>, unsigned char, _Constant int)">;
  def vminmaxss_round_mask : X86Builtin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Constant int, _Vector<4, float>, unsigned char, _Constant int)">;
  def vcvtbf162ibs128 : X86Builtin<"_Vector<8, unsigned short>(_Vector<8, __bf16>)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vcvtbf162ibs256 : X86Builtin<"_Vector<16, unsigned short>(_Vector<16, __bf16>)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vcvtbf162ibs512 : X86Builtin<"_Vector<32, unsigned short>(_Vector<32, __bf16>)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vcvtbf162iubs128 : X86Builtin<"_Vector<8, unsigned short>(_Vector<8, __bf16>)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vcvtbf162iubs256 : X86Builtin<"_Vector<16, unsigned short>(_Vector<16, __bf16>)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vcvtbf162iubs512 : X86Builtin<"_Vector<32, unsigned short>(_Vector<32, __bf16>)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vcvtph2ibs128_mask : X86Builtin<"_Vector<8, unsigned short>(_Vector<8, _Float16>, _Vector<8, unsigned short>, unsigned char)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vcvtph2ibs256_mask : X86Builtin<"_Vector<16, unsigned short>(_Vector<16, _Float16>, _Vector<16, unsigned short>, unsigned short)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vcvtph2ibs512_mask : X86Builtin<"_Vector<32, unsigned short>(_Vector<32, _Float16>, _Vector<32, unsigned short>, unsigned int, _Constant int)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vcvtph2iubs128_mask : X86Builtin<"_Vector<8, unsigned short>(_Vector<8, _Float16>, _Vector<8, unsigned short>, unsigned char)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vcvtph2iubs256_mask : X86Builtin<"_Vector<16, unsigned short>(_Vector<16, _Float16>, _Vector<16, unsigned short>, unsigned short)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vcvtph2iubs512_mask : X86Builtin<"_Vector<32, unsigned short>(_Vector<32, _Float16>, _Vector<32, unsigned short>, unsigned int, _Constant int)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vcvtps2ibs128_mask : X86Builtin<"_Vector<4, unsigned int>(_Vector<4, float>, _Vector<4, unsigned int>, unsigned char)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vcvtps2ibs256_mask : X86Builtin<"_Vector<8, unsigned int>(_Vector<8, float>, _Vector<8, unsigned int>, unsigned char)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vcvtps2ibs512_mask : X86Builtin<"_Vector<16, unsigned int>(_Vector<16, float>, _Vector<16, unsigned int>, unsigned short, _Constant int)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vcvtps2iubs128_mask : X86Builtin<"_Vector<4, unsigned int>(_Vector<4, float>, _Vector<4, unsigned int>, unsigned char)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vcvtps2iubs256_mask : X86Builtin<"_Vector<8, unsigned int>(_Vector<8, float>, _Vector<8, unsigned int>, unsigned char)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vcvtps2iubs512_mask : X86Builtin<"_Vector<16, unsigned int>(_Vector<16, float>, _Vector<16, unsigned int>, unsigned short, _Constant int)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vcvttbf162ibs128 : X86Builtin<"_Vector<8, unsigned short>(_Vector<8, __bf16>)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vcvttbf162ibs256 : X86Builtin<"_Vector<16, unsigned short>(_Vector<16, __bf16>)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vcvttbf162ibs512 : X86Builtin<"_Vector<32, unsigned short>(_Vector<32, __bf16>)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vcvttbf162iubs128 : X86Builtin<"_Vector<8, unsigned short>(_Vector<8, __bf16>)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vcvttbf162iubs256 : X86Builtin<"_Vector<16, unsigned short>(_Vector<16, __bf16>)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vcvttbf162iubs512 : X86Builtin<"_Vector<32, unsigned short>(_Vector<32, __bf16>)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vcvttph2ibs128_mask : X86Builtin<"_Vector<8, unsigned short>(_Vector<8, _Float16>, _Vector<8, unsigned short>, unsigned char)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vcvttph2ibs256_mask : X86Builtin<"_Vector<16, unsigned short>(_Vector<16, _Float16>, _Vector<16, unsigned short>, unsigned short)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vcvttph2ibs512_mask : X86Builtin<"_Vector<32, unsigned short>(_Vector<32, _Float16>, _Vector<32, unsigned short>, unsigned int, _Constant int)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vcvttph2iubs128_mask : X86Builtin<"_Vector<8, unsigned short>(_Vector<8, _Float16>, _Vector<8, unsigned short>, unsigned char)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vcvttph2iubs256_mask : X86Builtin<"_Vector<16, unsigned short>(_Vector<16, _Float16>, _Vector<16, unsigned short>, unsigned short)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vcvttph2iubs512_mask : X86Builtin<"_Vector<32, unsigned short>(_Vector<32, _Float16>, _Vector<32, unsigned short>, unsigned int, _Constant int)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vcvttps2ibs128_mask : X86Builtin<"_Vector<4, unsigned int>(_Vector<4, float>, _Vector<4, unsigned int>, unsigned char)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vcvttps2ibs256_mask : X86Builtin<"_Vector<8, unsigned int>(_Vector<8, float>, _Vector<8, unsigned int>, unsigned char)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vcvttps2ibs512_mask : X86Builtin<"_Vector<16, unsigned int>(_Vector<16, float>, _Vector<16, unsigned int>, unsigned short, _Constant int)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vcvttps2iubs128_mask : X86Builtin<"_Vector<4, unsigned int>(_Vector<4, float>, _Vector<4, unsigned int>, unsigned char)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vcvttps2iubs256_mask : X86Builtin<"_Vector<8, unsigned int>(_Vector<8, float>, _Vector<8, unsigned int>, unsigned char)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vcvttps2iubs512_mask : X86Builtin<"_Vector<16, unsigned int>(_Vector<16, float>, _Vector<16, unsigned int>, unsigned short, _Constant int)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vcvt2ps2phx128_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<4, float>, _Vector<4, float>, _Vector<8, _Float16>, unsigned char)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vcvt2ps2phx256_mask : X86Builtin<"_Vector<16, _Float16>(_Vector<8, float>, _Vector<8, float>, _Vector<16, _Float16>, unsigned short)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vcvt2ps2phx512_mask : X86Builtin<"_Vector<32, _Float16>(_Vector<16, float>, _Vector<16, float>, _Vector<32, _Float16>, unsigned int, _Constant int)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vcvtbiasph2bf8_128_mask : X86Builtin<"_Vector<16, char>(_Vector<16, char>, _Vector<8, _Float16>, _Vector<16, char>, unsigned char)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vcvtbiasph2bf8_256_mask : X86Builtin<"_Vector<16, char>(_Vector<32, char>, _Vector<16, _Float16>, _Vector<16, char>, unsigned short)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vcvtbiasph2bf8_512_mask : X86Builtin<"_Vector<32, char>(_Vector<64, char>, _Vector<32, _Float16>, _Vector<32, char>, unsigned int)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vcvtbiasph2bf8s_128_mask : X86Builtin<"_Vector<16, char>(_Vector<16, char>, _Vector<8, _Float16>, _Vector<16, char>, unsigned char)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vcvtbiasph2bf8s_256_mask : X86Builtin<"_Vector<16, char>(_Vector<32, char>, _Vector<16, _Float16>, _Vector<16, char>, unsigned short)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vcvtbiasph2bf8s_512_mask : X86Builtin<"_Vector<32, char>(_Vector<64, char>, _Vector<32, _Float16>, _Vector<32, char>, unsigned int)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vcvtbiasph2hf8_128_mask : X86Builtin<"_Vector<16, char>(_Vector<16, char>, _Vector<8, _Float16>, _Vector<16, char>, unsigned char)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vcvtbiasph2hf8_256_mask : X86Builtin<"_Vector<16, char>(_Vector<32, char>, _Vector<16, _Float16>, _Vector<16, char>, unsigned short)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vcvtbiasph2hf8_512_mask : X86Builtin<"_Vector<32, char>(_Vector<64, char>, _Vector<32, _Float16>, _Vector<32, char>, unsigned int)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vcvtbiasph2hf8s_128_mask : X86Builtin<"_Vector<16, char>(_Vector<16, char>, _Vector<8, _Float16>, _Vector<16, char>, unsigned char)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vcvtbiasph2hf8s_256_mask : X86Builtin<"_Vector<16, char>(_Vector<32, char>, _Vector<16, _Float16>, _Vector<16, char>, unsigned short)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vcvtbiasph2hf8s_512_mask : X86Builtin<"_Vector<32, char>(_Vector<64, char>, _Vector<32, _Float16>, _Vector<32, char>, unsigned int)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vcvt2ph2bf8_128 : X86Builtin<"_Vector<16, char>(_Vector<8, _Float16>, _Vector<8, _Float16>)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vcvt2ph2bf8_256 : X86Builtin<"_Vector<32, char>(_Vector<16, _Float16>, _Vector<16, _Float16>)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vcvt2ph2bf8_512 : X86Builtin<"_Vector<64, char>(_Vector<32, _Float16>, _Vector<32, _Float16>)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vcvt2ph2bf8s_128 : X86Builtin<"_Vector<16, char>(_Vector<8, _Float16>, _Vector<8, _Float16>)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vcvt2ph2bf8s_256 : X86Builtin<"_Vector<32, char>(_Vector<16, _Float16>, _Vector<16, _Float16>)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vcvt2ph2bf8s_512 : X86Builtin<"_Vector<64, char>(_Vector<32, _Float16>, _Vector<32, _Float16>)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vcvt2ph2hf8_128 : X86Builtin<"_Vector<16, char>(_Vector<8, _Float16>, _Vector<8, _Float16>)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vcvt2ph2hf8_256 : X86Builtin<"_Vector<32, char>(_Vector<16, _Float16>, _Vector<16, _Float16>)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vcvt2ph2hf8_512 : X86Builtin<"_Vector<64, char>(_Vector<32, _Float16>, _Vector<32, _Float16>)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vcvt2ph2hf8s_128 : X86Builtin<"_Vector<16, char>(_Vector<8, _Float16>, _Vector<8, _Float16>)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vcvt2ph2hf8s_256 : X86Builtin<"_Vector<32, char>(_Vector<16, _Float16>, _Vector<16, _Float16>)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vcvt2ph2hf8s_512 : X86Builtin<"_Vector<64, char>(_Vector<32, _Float16>, _Vector<32, _Float16>)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vcvthf8_2ph128_mask : X86Builtin<"_Vector<8, _Float16>(_Vector<16, char>, _Vector<8, _Float16>, unsigned char)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vcvthf8_2ph256_mask : X86Builtin<"_Vector<16, _Float16>(_Vector<16, char>, _Vector<16, _Float16>, unsigned short)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vcvthf8_2ph512_mask : X86Builtin<"_Vector<32, _Float16>(_Vector<32, char>, _Vector<32, _Float16>, unsigned int)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vcvtph2bf8_128_mask : X86Builtin<"_Vector<16, char>(_Vector<8, _Float16>, _Vector<16, char>, unsigned char)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vcvtph2bf8_256_mask : X86Builtin<"_Vector<16, char>(_Vector<16, _Float16>, _Vector<16, char>, unsigned short)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vcvtph2bf8_512_mask : X86Builtin<"_Vector<32, char>(_Vector<32, _Float16>, _Vector<32, char>, unsigned int)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vcvtph2bf8s_128_mask : X86Builtin<"_Vector<16, char>(_Vector<8, _Float16>, _Vector<16, char>, unsigned char)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vcvtph2bf8s_256_mask : X86Builtin<"_Vector<16, char>(_Vector<16, _Float16>, _Vector<16, char>, unsigned short)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vcvtph2bf8s_512_mask : X86Builtin<"_Vector<32, char>(_Vector<32, _Float16>, _Vector<32, char>, unsigned int)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vcvtph2hf8_128_mask : X86Builtin<"_Vector<16, char>(_Vector<8, _Float16>, _Vector<16, char>, unsigned char)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vcvtph2hf8_256_mask : X86Builtin<"_Vector<16, char>(_Vector<16, _Float16>, _Vector<16, char>, unsigned short)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vcvtph2hf8_512_mask : X86Builtin<"_Vector<32, char>(_Vector<32, _Float16>, _Vector<32, char>, unsigned int)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vcvtph2hf8s_128_mask : X86Builtin<"_Vector<16, char>(_Vector<8, _Float16>, _Vector<16, char>, unsigned char)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vcvtph2hf8s_256_mask : X86Builtin<"_Vector<16, char>(_Vector<16, _Float16>, _Vector<16, char>, unsigned short)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vcvtph2hf8s_512_mask : X86Builtin<"_Vector<32, char>(_Vector<32, _Float16>, _Vector<32, char>, unsigned int)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def loadsbf16128_mask : X86Builtin<"_Vector<8, __bf16>(_Vector<8, __bf16 const *>, _Vector<8, __bf16>, unsigned char)">;
  def storesbf16128_mask : X86Builtin<"void(_Vector<8, __bf16 *>, _Vector<8, __bf16>, unsigned char)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vaddbf16128 : X86Builtin<"_Vector<8, __bf16>(_Vector<8, __bf16>, _Vector<8, __bf16>)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vaddbf16256 : X86Builtin<"_Vector<16, __bf16>(_Vector<16, __bf16>, _Vector<16, __bf16>)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vaddbf16512 : X86Builtin<"_Vector<32, __bf16>(_Vector<32, __bf16>, _Vector<32, __bf16>)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vdivbf16128 : X86Builtin<"_Vector<8, __bf16>(_Vector<8, __bf16>, _Vector<8, __bf16>)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vdivbf16256 : X86Builtin<"_Vector<16, __bf16>(_Vector<16, __bf16>, _Vector<16, __bf16>)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vdivbf16512 : X86Builtin<"_Vector<32, __bf16>(_Vector<32, __bf16>, _Vector<32, __bf16>)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vmaxbf16128 : X86Builtin<"_Vector<8, __bf16>(_Vector<8, __bf16>, _Vector<8, __bf16>)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vmaxbf16256 : X86Builtin<"_Vector<16, __bf16>(_Vector<16, __bf16>, _Vector<16, __bf16>)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vmaxbf16512 : X86Builtin<"_Vector<32, __bf16>(_Vector<32, __bf16>, _Vector<32, __bf16>)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vminbf16128 : X86Builtin<"_Vector<8, __bf16>(_Vector<8, __bf16>, _Vector<8, __bf16>)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vminbf16256 : X86Builtin<"_Vector<16, __bf16>(_Vector<16, __bf16>, _Vector<16, __bf16>)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vminbf16512 : X86Builtin<"_Vector<32, __bf16>(_Vector<32, __bf16>, _Vector<32, __bf16>)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vmulbf16128 : X86Builtin<"_Vector<8, __bf16>(_Vector<8, __bf16>, _Vector<8, __bf16>)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vmulbf16256 : X86Builtin<"_Vector<16, __bf16>(_Vector<16, __bf16>, _Vector<16, __bf16>)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vmulbf16512 : X86Builtin<"_Vector<32, __bf16>(_Vector<32, __bf16>, _Vector<32, __bf16>)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vsubbf16128 : X86Builtin<"_Vector<8, __bf16>(_Vector<8, __bf16>, _Vector<8, __bf16>)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vsubbf16256 : X86Builtin<"_Vector<16, __bf16>(_Vector<16, __bf16>, _Vector<16, __bf16>)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vsubbf16512 : X86Builtin<"_Vector<32, __bf16>(_Vector<32, __bf16>, _Vector<32, __bf16>)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in
  foreach Cmp = ["eq", "lt", "le", "gt", "ge", "neq"] in
    def vcomisbf16#Cmp : X86Builtin<"int(_Vector<8, __bf16>, _Vector<8, __bf16>)">;

let Features = "avx10.2-512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vcmpbf16512_mask : X86Builtin<"unsigned int(_Vector<32, __bf16>, _Vector<32, __bf16>, _Constant int, unsigned int)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vcmpbf16256_mask : X86Builtin<"unsigned short(_Vector<16, __bf16>, _Vector<16, __bf16>, _Constant int, unsigned short)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vcmpbf16128_mask : X86Builtin<"unsigned char(_Vector<8, __bf16>, _Vector<8, __bf16>, _Constant int, unsigned char)">;
  def vfpclassbf16128_mask : X86Builtin<"unsigned char(_Vector<8, __bf16>, _Constant int, unsigned char)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vfpclassbf16256_mask : X86Builtin<"unsigned short(_Vector<16, __bf16>, _Constant int, unsigned short)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vfpclassbf16512_mask : X86Builtin<"unsigned int(_Vector<32, __bf16>, _Constant int, unsigned int)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vscalefbf16128_mask : X86Builtin<"_Vector<8, __bf16>(_Vector<8, __bf16>, _Vector<8, __bf16>, _Vector<8, __bf16>, unsigned char)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vscalefbf16256_mask : X86Builtin<"_Vector<16, __bf16>(_Vector<16, __bf16>, _Vector<16, __bf16>, _Vector<16, __bf16>, unsigned short)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vscalefbf16512_mask : X86Builtin<"_Vector<32, __bf16>(_Vector<32, __bf16>, _Vector<32, __bf16>, _Vector<32, __bf16>, unsigned int)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vrcpbf16128_mask : X86Builtin<"_Vector<8, __bf16>(_Vector<8, __bf16>, _Vector<8, __bf16>, unsigned char)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vrcpbf16256_mask : X86Builtin<"_Vector<16, __bf16>(_Vector<16, __bf16>, _Vector<16, __bf16>, unsigned short)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vrcpbf16512_mask : X86Builtin<"_Vector<32, __bf16>(_Vector<32, __bf16>, _Vector<32, __bf16>, unsigned int)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vgetexpbf16128_mask : X86Builtin<"_Vector<8, __bf16>(_Vector<8, __bf16>, _Vector<8, __bf16>, unsigned char)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vgetexpbf16256_mask : X86Builtin<"_Vector<16, __bf16>(_Vector<16, __bf16>, _Vector<16, __bf16>, unsigned short)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vgetexpbf16512_mask : X86Builtin<"_Vector<32, __bf16>(_Vector<32, __bf16>, _Vector<32, __bf16>, unsigned int)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vrsqrtbf16128_mask : X86Builtin<"_Vector<8, __bf16>(_Vector<8, __bf16>, _Vector<8, __bf16>, unsigned char)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vrsqrtbf16256_mask : X86Builtin<"_Vector<16, __bf16>(_Vector<16, __bf16>, _Vector<16, __bf16>, unsigned short)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vrsqrtbf16512_mask : X86Builtin<"_Vector<32, __bf16>(_Vector<32, __bf16>, _Vector<32, __bf16>, unsigned int)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vreducebf16128_mask : X86Builtin<"_Vector<8, __bf16>(_Vector<8, __bf16>, _Constant int, _Vector<8, __bf16>, unsigned char)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vreducebf16256_mask : X86Builtin<"_Vector<16, __bf16>(_Vector<16, __bf16>, _Constant int, _Vector<16, __bf16>, unsigned short)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vreducebf16512_mask : X86Builtin<"_Vector<32, __bf16>(_Vector<32, __bf16>, _Constant int, _Vector<32, __bf16>, unsigned int)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vrndscalebf16_128_mask : X86Builtin<"_Vector<8, __bf16>(_Vector<8, __bf16>, _Constant int, _Vector<8, __bf16>, unsigned char)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vrndscalebf16_256_mask : X86Builtin<"_Vector<16, __bf16>(_Vector<16, __bf16>, _Constant int, _Vector<16, __bf16>, unsigned short)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vrndscalebf16_mask : X86Builtin<"_Vector<32, __bf16>(_Vector<32, __bf16>, _Constant int, _Vector<32, __bf16>, unsigned int)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vgetmantbf16128_mask : X86Builtin<"_Vector<8, __bf16>(_Vector<8, __bf16>, _Constant int, _Vector<8, __bf16>, unsigned char)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vgetmantbf16256_mask : X86Builtin<"_Vector<16, __bf16>(_Vector<16, __bf16>, _Constant int, _Vector<16, __bf16>, unsigned short)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vgetmantbf16512_mask : X86Builtin<"_Vector<32, __bf16>(_Vector<32, __bf16>, _Constant int, _Vector<32, __bf16>, unsigned int)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vsqrtbf16 : X86Builtin<"_Vector<8, __bf16>(_Vector<8, __bf16>)">;
}

let Features = "avx10.2-256", Attributes = [NoThrow, Const, RequiredVectorWidth<256>] in {
  def vsqrtbf16256 : X86Builtin<"_Vector<16, __bf16>(_Vector<16, __bf16>)">;
}

let Features = "avx10.2-512", Attributes = [NoThrow, Const, RequiredVectorWidth<512>] in {
  def vsqrtbf16512 : X86Builtin<"_Vector<32, __bf16>(_Vector<32, __bf16>)">;
}
