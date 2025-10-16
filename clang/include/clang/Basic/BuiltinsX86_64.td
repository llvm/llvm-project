//===--- BuiltinsX86_64.td - X86-64 Builtin function database ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the X86-64-specific builtin function database.
//
//===----------------------------------------------------------------------===//

include "clang/Basic/BuiltinsX86Base.td"

let Header = "intrin.h", Languages = "ALL_MS_LANGUAGES", Attributes = [NoThrow, RequireDeclaration] in {
  def _BitScanForward64 : X86LibBuiltin<"unsigned char(msuint32_t *, unsigned long long int)">;
  def _BitScanReverse64 : X86LibBuiltin<"unsigned char(msuint32_t *, unsigned long long int)">;
}

let Header = "intrin.h", Languages = "ALL_MS_LANGUAGES", Attributes = [NoThrow, Const, RequireDeclaration] in {
  def __mulh : X86LibBuiltin<"long long int(long long int, long long int)">;
  def __umulh : X86LibBuiltin<"unsigned long long int(unsigned long long int, unsigned long long int)">;
  def _mul128 : X86LibBuiltin<"long long int(long long int, long long int, long long int *)">;
  def _umul128 : X86LibBuiltin<"unsigned long long int(unsigned long long int, unsigned long long int, unsigned long long int *)">;
}

let Header = "intrin.h", Languages = "ALL_MS_LANGUAGES", Attributes = [NoThrow, RequireDeclaration] in {
  def __faststorefence : X86LibBuiltin<"void()">;
}

let Header = "intrin.h", Languages = "ALL_MS_LANGUAGES", Attributes = [NoThrow, Const, RequireDeclaration] in {
  def __shiftleft128 : X86LibBuiltin<"unsigned long long int(unsigned long long int, unsigned long long int, unsigned char)">;
  def __shiftright128 : X86LibBuiltin<"unsigned long long int(unsigned long long int, unsigned long long int, unsigned char)">;
}

let Features = "cx16", Header = "intrin.h", Languages = "ALL_MS_LANGUAGES", Attributes = [NoThrow, RequireDeclaration] in {
  def _InterlockedCompareExchange128 : X86LibBuiltin<"unsigned char(long long int volatile *, long long int, long long int, long long int *)">;
}

let Attributes = [NoThrow] in {
  def readeflags_u64 : X86Builtin<"unsigned long long int()">;
  def writeeflags_u64 : X86Builtin<"void(unsigned long long int)">;
}

let Features = "sse", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def cvtss2si64 : X86Builtin<"long long int(_Vector<4, float>)">;
  def cvttss2si64 : X86Builtin<"long long int(_Vector<4, float>)">;
}

let Features = "sse2", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def cvtsd2si64 : X86Builtin<"long long int(_Vector<2, double>)">;
  def cvttsd2si64 : X86Builtin<"long long int(_Vector<2, double>)">;
}

let Features = "sse2", Attributes = [NoThrow] in {
  def movnti64 : X86Builtin<"void(long long int *, long long int)">;
}

let Features = "sse4.1", Attributes = [NoThrow, Const, Constexpr, RequiredVectorWidth<128>] in {
  def vec_set_v2di : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int>, long long int, _Constant int)">;
}

let Features = "crc32", Attributes = [NoThrow, Const] in {
  def crc32di : X86Builtin<"unsigned long long int(unsigned long long int, unsigned long long int)">;
}

let Features = "avx", Attributes = [NoThrow, Const, Constexpr, RequiredVectorWidth<256>] in {
  def vec_ext_v4di : X86Builtin<"long long int(_Vector<4, long long int>, _Constant int)">;
  def vec_set_v4di : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int>, long long int, _Constant int)">;
}

let Features = "fsgsbase", Attributes = [NoThrow] in {
  def rdfsbase32 : X86Builtin<"unsigned int()">;
  def rdfsbase64 : X86Builtin<"unsigned long long int()">;
  def rdgsbase32 : X86Builtin<"unsigned int()">;
  def rdgsbase64 : X86Builtin<"unsigned long long int()">;
  def wrfsbase32 : X86Builtin<"void(unsigned int)">;
  def wrfsbase64 : X86Builtin<"void(unsigned long long int)">;
  def wrgsbase32 : X86Builtin<"void(unsigned int)">;
  def wrgsbase64 : X86Builtin<"void(unsigned long long int)">;
}

let Features = "fxsr", Attributes = [NoThrow] in {
  def fxrstor64 : X86Builtin<"void(void *)">;
  def fxsave64 : X86Builtin<"void(void *)">;
}

let Features = "xsave", Attributes = [NoThrow] in {
  def xsave64 : X86Builtin<"void(void *, unsigned long long int)">;
  def xrstor64 : X86Builtin<"void(void *, unsigned long long int)">;
}

let Features = "xsaveopt", Attributes = [NoThrow] in {
  def xsaveopt64 : X86Builtin<"void(void *, unsigned long long int)">;
}

let Features = "xsaves", Attributes = [NoThrow] in {
  def xrstors64 : X86Builtin<"void(void *, unsigned long long int)">;
}

let Features = "xsavec", Attributes = [NoThrow] in {
  def xsavec64 : X86Builtin<"void(void *, unsigned long long int)">;
}

let Features = "xsaves", Attributes = [NoThrow] in {
  def xsaves64 : X86Builtin<"void(void *, unsigned long long int)">;
}

let Features = "shstk", Attributes = [NoThrow] in {
  def incsspq : X86Builtin<"void(unsigned long long int)">;
  def rdsspq : X86Builtin<"unsigned long long int(unsigned long long int)">;
  def wrssq : X86Builtin<"void(unsigned long long int, void *)">;
  def wrussq : X86Builtin<"void(unsigned long long int, void *)">;
}

let Attributes = [NoThrow, Constexpr] in {
  def addcarryx_u64 : X86Builtin<"unsigned char(unsigned char, unsigned long long int, unsigned long long int, unsigned long long int *)">;
  def subborrow_u64 : X86Builtin<"unsigned char(unsigned char, unsigned long long int, unsigned long long int, unsigned long long int *)">;
}

let Features = "rdrnd", Attributes = [NoThrow] in {
  def rdrand64_step : X86Builtin<"unsigned int(unsigned long long int *)">;
}

let Features = "rdseed", Attributes = [NoThrow] in {
  def rdseed64_step : X86Builtin<"unsigned int(unsigned long long int *)">;
}

let Features = "bmi", Attributes = [NoThrow, Const, Constexpr] in {
  def bextr_u64 : X86Builtin<"unsigned long long int(unsigned long long int, unsigned long long int)">;
}

let Attributes = [NoThrow, Const, Constexpr] in {
  def tzcnt_u64 : X86Builtin<"unsigned long long int(unsigned long long int)">;
  def lzcnt_u64 : X86Builtin<"unsigned long long int(unsigned long long int)">;
}

let Features = "bmi2", Attributes = [NoThrow, Const, Constexpr] in {
  def bzhi_di : X86Builtin<"unsigned long long int(unsigned long long int, unsigned long long int)">;
  def pdep_di : X86Builtin<"unsigned long long int(unsigned long long int, unsigned long long int)">;
  def pext_di : X86Builtin<"unsigned long long int(unsigned long long int, unsigned long long int)">;
}

let Features = "tbm", Attributes = [NoThrow, Const, Constexpr] in {
  def bextri_u64 : X86Builtin<"unsigned long long int(unsigned long long int, _Constant unsigned long long int)">;
}

let Features = "lwp", Attributes = [NoThrow] in {
  def lwpins64 : X86Builtin<"unsigned char(unsigned long long int, unsigned int, _Constant unsigned int)">;
  def lwpval64 : X86Builtin<"void(unsigned long long int, unsigned int, _Constant unsigned int)">;
}

let Features = "avx512f", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vcvtsd2si64 : X86Builtin<"long long int(_Vector<2, double>, _Constant int)">;
  def vcvtsd2usi64 : X86Builtin<"unsigned long long int(_Vector<2, double>, _Constant int)">;
  def vcvtss2si64 : X86Builtin<"long long int(_Vector<4, float>, _Constant int)">;
  def vcvtss2usi64 : X86Builtin<"unsigned long long int(_Vector<4, float>, _Constant int)">;
  def vcvttsd2si64 : X86Builtin<"long long int(_Vector<2, double>, _Constant int)">;
  def vcvttsd2usi64 : X86Builtin<"unsigned long long int(_Vector<2, double>, _Constant int)">;
  def vcvttss2si64 : X86Builtin<"long long int(_Vector<4, float>, _Constant int)">;
  def vcvttss2usi64 : X86Builtin<"unsigned long long int(_Vector<4, float>, _Constant int)">;
  def cvtsi2sd64 : X86Builtin<"_Vector<2, double>(_Vector<2, double>, long long int, _Constant int)">;
  def cvtsi2ss64 : X86Builtin<"_Vector<4, float>(_Vector<4, float>, long long int, _Constant int)">;
  def cvtusi2sd64 : X86Builtin<"_Vector<2, double>(_Vector<2, double>, unsigned long long int, _Constant int)">;
  def cvtusi2ss64 : X86Builtin<"_Vector<4, float>(_Vector<4, float>, unsigned long long int, _Constant int)">;
}

let Features = "avx512fp16", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vcvtsh2si64 : X86Builtin<"long long int(_Vector<8, _Float16>, _Constant int)">;
  def vcvtsh2usi64 : X86Builtin<"unsigned long long int(_Vector<8, _Float16>, _Constant int)">;
  def vcvtusi642sh : X86Builtin<"_Vector<8, _Float16>(_Vector<8, _Float16>, unsigned long long int, _Constant int)">;
  def vcvtsi642sh : X86Builtin<"_Vector<8, _Float16>(_Vector<8, _Float16>, long long int, _Constant int)">;
  def vcvttsh2si64 : X86Builtin<"long long int(_Vector<8, _Float16>, _Constant int)">;
  def vcvttsh2usi64 : X86Builtin<"unsigned long long int(_Vector<8, _Float16>, _Constant int)">;
}

let Features = "movdiri", Attributes = [NoThrow] in {
  def directstore_u64 : X86Builtin<"void(unsigned long int *, unsigned long int)">;
}

let Features = "avx10.2", Attributes = [NoThrow, Const, RequiredVectorWidth<128>] in {
  def vcvttsd2sis64 : X86Builtin<"long long int(_Vector<2, double>, _Constant int)">;
  def vcvttsd2usis64 : X86Builtin<"unsigned long long int(_Vector<2, double>, _Constant int)">;
  def vcvttss2sis64 : X86Builtin<"long long int(_Vector<4, float>, _Constant int)">;
  def vcvttss2usis64 : X86Builtin<"unsigned long long int(_Vector<4, float>, _Constant int)">;
}

let Features = "uintr", Attributes = [NoThrow] in {
  def clui : X86Builtin<"void()">;
  def stui : X86Builtin<"void()">;
  def testui : X86Builtin<"unsigned char()">;
  def senduipi : X86Builtin<"void(uint64_t)">;
}

let Features = "usermsr", Attributes = [NoThrow] in {
  def urdmsr : X86Builtin<"unsigned long long int(unsigned long long int)">;
  def uwrmsr : X86Builtin<"void(unsigned long long int, unsigned long long int)">;
}

let Features = "amx-tile", Attributes = [NoThrow] in {
  def tile_loadconfig_internal : X86Builtin<"void(void const *)">;
  def tileloadd64_internal : X86Builtin<"_Vector<256, int>(unsigned short, unsigned short, void const *, size_t)">;
}

let Features = "amx-movrs", Attributes = [NoThrow] in {
  def tileloaddrs64_internal : X86Builtin<"_Vector<256, int>(unsigned short, unsigned short, void const *, size_t)">;
}

let Features = "amx-tile", Attributes = [NoThrow] in {
  def tileloaddt164_internal : X86Builtin<"_Vector<256, int>(unsigned short, unsigned short, void const *, size_t)">;
}

let Features = "amx-movrs", Attributes = [NoThrow] in {
  def tileloaddrst164_internal : X86Builtin<"_Vector<256, int>(unsigned short, unsigned short, void const *, size_t)">;
}

let Features = "amx-int8", Attributes = [NoThrow] in {
  def tdpbssd_internal : X86Builtin<"_Vector<256, int>(unsigned short, unsigned short, unsigned short, _Vector<256, int>, _Vector<256, int>, _Vector<256, int>)">;
  def tdpbsud_internal : X86Builtin<"_Vector<256, int>(unsigned short, unsigned short, unsigned short, _Vector<256, int>, _Vector<256, int>, _Vector<256, int>)">;
  def tdpbusd_internal : X86Builtin<"_Vector<256, int>(unsigned short, unsigned short, unsigned short, _Vector<256, int>, _Vector<256, int>, _Vector<256, int>)">;
  def tdpbuud_internal : X86Builtin<"_Vector<256, int>(unsigned short, unsigned short, unsigned short, _Vector<256, int>, _Vector<256, int>, _Vector<256, int>)">;
}

let Features = "amx-tile", Attributes = [NoThrow] in {
  def tilestored64_internal : X86Builtin<"void(unsigned short, unsigned short, void *, size_t, _Vector<256, int>)">;
  def tilezero_internal : X86Builtin<"_Vector<256, int>(unsigned short, unsigned short)">;
}

let Features = "amx-bf16", Attributes = [NoThrow] in {
  def tdpbf16ps_internal : X86Builtin<"_Vector<256, int>(unsigned short, unsigned short, unsigned short, _Vector<256, int>, _Vector<256, int>, _Vector<256, int>)">;
}

let Features = "amx-fp16", Attributes = [NoThrow] in {
  def tdpfp16ps_internal : X86Builtin<"_Vector<256, int>(unsigned short, unsigned short, unsigned short, _Vector<256, int>, _Vector<256, int>, _Vector<256, int>)">;
}

let Features = "amx-complex", Attributes = [NoThrow] in {
  def tcmmimfp16ps_internal : X86Builtin<"_Vector<256, int>(unsigned short, unsigned short, unsigned short, _Vector<256, int>, _Vector<256, int>, _Vector<256, int>)">;
  def tcmmrlfp16ps_internal : X86Builtin<"_Vector<256, int>(unsigned short, unsigned short, unsigned short, _Vector<256, int>, _Vector<256, int>, _Vector<256, int>)">;
}

let Features = "amx-transpose", Attributes = [NoThrow] in {
  def t2rpntlvwz0_internal : X86Builtin<"void(unsigned short, unsigned short, unsigned short, _Vector<256, int *>, _Vector<256, int *>, void const *, size_t)">;
}

let Features = "amx-movrs,amx-transpose", Attributes = [NoThrow] in {
  def t2rpntlvwz0rs_internal : X86Builtin<"void(unsigned short, unsigned short, unsigned short, _Vector<256, int *>, _Vector<256, int *>, void const *, size_t)">;
}

let Features = "amx-transpose", Attributes = [NoThrow] in {
  def t2rpntlvwz0t1_internal : X86Builtin<"void(unsigned short, unsigned short, unsigned short, _Vector<256, int *>, _Vector<256, int *>, void const *, size_t)">;
}

let Features = "amx-movrs,amx-transpose", Attributes = [NoThrow] in {
  def t2rpntlvwz0rst1_internal : X86Builtin<"void(unsigned short, unsigned short, unsigned short, _Vector<256, int *>, _Vector<256, int *>, void const *, size_t)">;
}

let Features = "amx-transpose", Attributes = [NoThrow] in {
  def t2rpntlvwz1_internal : X86Builtin<"void(unsigned short, unsigned short, unsigned short, _Vector<256, int *>, _Vector<256, int *>, void const *, size_t)">;
}

let Features = "amx-movrs,amx-transpose", Attributes = [NoThrow] in {
  def t2rpntlvwz1rs_internal : X86Builtin<"void(unsigned short, unsigned short, unsigned short, _Vector<256, int *>, _Vector<256, int *>, void const *, size_t)">;
}

let Features = "amx-transpose", Attributes = [NoThrow] in {
  def t2rpntlvwz1t1_internal : X86Builtin<"void(unsigned short, unsigned short, unsigned short, _Vector<256, int *>, _Vector<256, int *>, void const *, size_t)">;
}

let Features = "amx-movrs,amx-transpose", Attributes = [NoThrow] in {
  def t2rpntlvwz1rst1_internal : X86Builtin<"void(unsigned short, unsigned short, unsigned short, _Vector<256, int *>, _Vector<256, int *>, void const *, size_t)">;
}

let Features = "amx-transpose", Attributes = [NoThrow] in {
  def ttransposed_internal : X86Builtin<"_Vector<256, int>(unsigned short, unsigned short, _Vector<256, int>)">;
}

let Features = "amx-bf16,amx-transpose", Attributes = [NoThrow] in {
  def ttdpbf16ps_internal : X86Builtin<"_Vector<256, int>(unsigned short, unsigned short, unsigned short, _Vector<256, int>, _Vector<256, int>, _Vector<256, int>)">;
}

let Features = "amx-fp16,amx-transpose", Attributes = [NoThrow] in {
  def ttdpfp16ps_internal : X86Builtin<"_Vector<256, int>(unsigned short, unsigned short, unsigned short, _Vector<256, int>, _Vector<256, int>, _Vector<256, int>)">;
}

let Features = "amx-complex,amx-transpose", Attributes = [NoThrow] in {
  def ttcmmimfp16ps_internal : X86Builtin<"_Vector<256, int>(unsigned short, unsigned short, unsigned short, _Vector<256, int>, _Vector<256, int>, _Vector<256, int>)">;
  def ttcmmrlfp16ps_internal : X86Builtin<"_Vector<256, int>(unsigned short, unsigned short, unsigned short, _Vector<256, int>, _Vector<256, int>, _Vector<256, int>)">;
  def tconjtcmmimfp16ps_internal : X86Builtin<"_Vector<256, int>(unsigned short, unsigned short, unsigned short, _Vector<256, int>, _Vector<256, int>, _Vector<256, int>)">;
  def tconjtfp16_internal : X86Builtin<"_Vector<256, int>(unsigned short, unsigned short, _Vector<256, int>)">;
}

let Features = "amx-avx512,avx10.2", Attributes = [NoThrow] in {
  def tcvtrowd2ps_internal : X86Builtin<"_Vector<16, float>(unsigned short, unsigned short, _Vector<256, int>, unsigned int)">;
  def tcvtrowps2bf16h_internal : X86Builtin<"_Vector<32, __bf16>(unsigned short, unsigned short, _Vector<256, int>, unsigned int)">;
  def tcvtrowps2bf16l_internal : X86Builtin<"_Vector<32, __bf16>(unsigned short, unsigned short, _Vector<256, int>, unsigned int)">;
  def tcvtrowps2phh_internal : X86Builtin<"_Vector<32, _Float16>(unsigned short, unsigned short, _Vector<256, int>, unsigned int)">;
  def tcvtrowps2phl_internal : X86Builtin<"_Vector<32, _Float16>(unsigned short, unsigned short, _Vector<256, int>, unsigned int)">;
  def tilemovrow_internal : X86Builtin<"_Vector<16, int>(unsigned short, unsigned short, _Vector<256, int>, unsigned int)">;
}

let Features = "amx-tf32", Attributes = [NoThrow] in {
  def tmmultf32ps_internal : X86Builtin<"_Vector<256, int>(unsigned short, unsigned short, unsigned short, _Vector<256, int>, _Vector<256, int>, _Vector<256, int>)">;
}

let Features = "amx-tf32,amx-transpose", Attributes = [NoThrow] in {
  def ttmmultf32ps_internal : X86Builtin<"_Vector<256, int>(unsigned short, unsigned short, unsigned short, _Vector<256, int>, _Vector<256, int>, _Vector<256, int>)">;
}

let Features = "amx-fp8", Attributes = [NoThrow] in {
  def tdpbf8ps_internal : X86Builtin<"_Vector<256, int>(unsigned short, unsigned short, unsigned short, _Vector<256, int>, _Vector<256, int>, _Vector<256, int>)">;
  def tdpbhf8ps_internal : X86Builtin<"_Vector<256, int>(unsigned short, unsigned short, unsigned short, _Vector<256, int>, _Vector<256, int>, _Vector<256, int>)">;
  def tdphbf8ps_internal : X86Builtin<"_Vector<256, int>(unsigned short, unsigned short, unsigned short, _Vector<256, int>, _Vector<256, int>, _Vector<256, int>)">;
  def tdphf8ps_internal : X86Builtin<"_Vector<256, int>(unsigned short, unsigned short, unsigned short, _Vector<256, int>, _Vector<256, int>, _Vector<256, int>)">;
}

let Features = "amx-tile", Attributes = [NoThrow] in {
  def tile_loadconfig : X86Builtin<"void(void const *)">;
  def tile_storeconfig : X86Builtin<"void(void const *)">;
  def tilerelease : X86Builtin<"void()">;
  def tilezero : X86Builtin<"void(unsigned char)">;
}

let Features = "amx-movrs,amx-transpose", Attributes = [NoThrow] in {
  def t2rpntlvwz0rs : X86Builtin<"void(_Constant unsigned char, void const *, size_t)">;
  def t2rpntlvwz0rst1 : X86Builtin<"void(_Constant unsigned char, void const *, size_t)">;
  def t2rpntlvwz1rs : X86Builtin<"void(_Constant unsigned char, void const *, size_t)">;
  def t2rpntlvwz1rst1 : X86Builtin<"void(_Constant unsigned char, void const *, size_t)">;
}

let Features = "amx-movrs", Attributes = [NoThrow] in {
  def tileloaddrs64 : X86Builtin<"void(_Constant unsigned char, void const *, size_t)">;
  def tileloaddrst164 : X86Builtin<"void(_Constant unsigned char, void const *, size_t)">;
}

let Features = "amx-tile", Attributes = [NoThrow] in {
  def tileloadd64 : X86Builtin<"void(_Constant unsigned char, void const *, size_t)">;
  def tileloaddt164 : X86Builtin<"void(_Constant unsigned char, void const *, size_t)">;
  def tilestored64 : X86Builtin<"void(_Constant unsigned char, void *, size_t)">;
}

let Features = "amx-int8", Attributes = [NoThrow] in {
  def tdpbssd : X86Builtin<"void(_Constant unsigned char, _Constant unsigned char, _Constant unsigned char)">;
  def tdpbsud : X86Builtin<"void(_Constant unsigned char, _Constant unsigned char, _Constant unsigned char)">;
  def tdpbusd : X86Builtin<"void(_Constant unsigned char, _Constant unsigned char, _Constant unsigned char)">;
  def tdpbuud : X86Builtin<"void(_Constant unsigned char, _Constant unsigned char, _Constant unsigned char)">;
}

let Features = "amx-bf16", Attributes = [NoThrow] in {
  def tdpbf16ps : X86Builtin<"void(_Constant unsigned char, _Constant unsigned char, _Constant unsigned char)">;
}

let Features = "ptwrite", Attributes = [NoThrow] in {
  def ptwrite64 : X86Builtin<"void(unsigned long long int)">;
}

let Features = "amx-complex", Attributes = [NoThrow] in {
  def tcmmimfp16ps : X86Builtin<"void(_Constant unsigned char, _Constant unsigned char, _Constant unsigned char)">;
  def tcmmrlfp16ps : X86Builtin<"void(_Constant unsigned char, _Constant unsigned char, _Constant unsigned char)">;
}

let Features = "amx-transpose", Attributes = [NoThrow] in {
  def t2rpntlvwz0 : X86Builtin<"void(_Constant unsigned char, void const *, size_t)">;
  def t2rpntlvwz0t1 : X86Builtin<"void(_Constant unsigned char, void const *, size_t)">;
  def t2rpntlvwz1 : X86Builtin<"void(_Constant unsigned char, void const *, size_t)">;
  def t2rpntlvwz1t1 : X86Builtin<"void(_Constant unsigned char, void const *, size_t)">;
  def ttransposed : X86Builtin<"void(_Constant unsigned char, _Constant unsigned char)">;
}

let Features = "amx-bf16,amx-transpose", Attributes = [NoThrow] in {
  def ttdpbf16ps : X86Builtin<"void(_Constant unsigned char, _Constant unsigned char, _Constant unsigned char)">;
}

let Features = "amx-fp16,amx-transpose", Attributes = [NoThrow] in {
  def ttdpfp16ps : X86Builtin<"void(_Constant unsigned char, _Constant unsigned char, _Constant unsigned char)">;
}

let Features = "amx-complex,amx-transpose", Attributes = [NoThrow] in {
  def ttcmmimfp16ps : X86Builtin<"void(_Constant unsigned char, _Constant unsigned char, _Constant unsigned char)">;
  def ttcmmrlfp16ps : X86Builtin<"void(_Constant unsigned char, _Constant unsigned char, _Constant unsigned char)">;
  def tconjtcmmimfp16ps : X86Builtin<"void(_Constant unsigned char, _Constant unsigned char, _Constant unsigned char)">;
  def tconjtfp16 : X86Builtin<"void(_Constant unsigned char, _Constant unsigned char)">;
}

let Features = "amx-avx512,avx10.2", Attributes = [NoThrow] in {
  def tcvtrowd2ps : X86Builtin<"_Vector<16, float>(_Constant unsigned char, unsigned int)">;
  def tcvtrowps2bf16h : X86Builtin<"_Vector<32, __bf16>(_Constant unsigned char, unsigned int)">;
  def tcvtrowps2bf16l : X86Builtin<"_Vector<32, __bf16>(_Constant unsigned char, unsigned int)">;
  def tcvtrowps2phh : X86Builtin<"_Vector<32, _Float16>(_Constant unsigned char, unsigned int)">;
  def tcvtrowps2phl : X86Builtin<"_Vector<32, _Float16>(_Constant unsigned char, unsigned int)">;
  def tilemovrow : X86Builtin<"_Vector<16, int>(_Constant unsigned char, unsigned int)">;
}

let Features = "amx-fp16", Attributes = [NoThrow] in {
  def tdpfp16ps : X86Builtin<"void(_Constant unsigned char, _Constant unsigned char, _Constant unsigned char)">;
}

let Features = "amx-fp8", Attributes = [NoThrow] in {
  def tdpbf8ps : X86Builtin<"void(_Constant unsigned char, unsigned _Constant char, unsigned _Constant char)">;
  def tdpbhf8ps : X86Builtin<"void(_Constant unsigned char, unsigned _Constant char, unsigned _Constant char)">;
  def tdphbf8ps : X86Builtin<"void(_Constant unsigned char, unsigned _Constant char, unsigned _Constant char)">;
  def tdphf8ps : X86Builtin<"void(_Constant unsigned char, unsigned _Constant char, unsigned _Constant char)">;
}

let Features = "amx-tf32", Attributes = [NoThrow] in {
  def tmmultf32ps : X86Builtin<"void(_Constant unsigned char, _Constant unsigned char, _Constant unsigned char)">;
}

let Features = "amx-tf32,amx-transpose", Attributes = [NoThrow] in {
  def ttmmultf32ps : X86Builtin<"void(_Constant unsigned char, _Constant unsigned char, _Constant unsigned char)">;
}

let Features = "prefetchi", Attributes = [NoThrow, Const] in {
  def prefetchi : X86Builtin<"void(void const *, unsigned int)">;
}

let Features = "cmpccxadd", Attributes = [NoThrow] in {
  def cmpccxadd32 : X86Builtin<"signed int(void *, signed int, signed int, _Constant int)">;
  def cmpccxadd64 : X86Builtin<"signed long long int(signed long long int *, signed long long int, signed long long int, _Constant int)">;
}

let Features = "raoint", Attributes = [NoThrow] in {
  def aadd64 : X86Builtin<"void(void *, signed long long int)">;
  def aand64 : X86Builtin<"void(void *, signed long long int)">;
  def aor64 : X86Builtin<"void(void *, signed long long int)">;
  def axor64 : X86Builtin<"void(void *, signed long long int)">;
}

let Features = "movrs", Attributes = [NoThrow] in {
  def movrsqi : X86Builtin<"signed char(void const *)">;
  def movrshi : X86Builtin<"signed short(void const *)">;
  def movrssi : X86Builtin<"signed int(void const *)">;
  def movrsdi : X86Builtin<"signed long long int(void const *)">;
}

let Features = "movrs,avx10.2", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vmovrsb128 : X86Builtin<"_Vector<16, char>(_Vector<16, char const *>)">;
}

let Features = "movrs,avx10.2", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vmovrsb256 : X86Builtin<"_Vector<32, char>(_Vector<32, char const *>)">;
}

let Features = "movrs,avx10.2", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vmovrsb512 : X86Builtin<"_Vector<64, char>(_Vector<64, char const *>)">;
}

let Features = "movrs,avx10.2", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vmovrsd128 : X86Builtin<"_Vector<4, int>(_Vector<4, int const *>)">;
}

let Features = "movrs,avx10.2", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vmovrsd256 : X86Builtin<"_Vector<8, int>(_Vector<8, int const *>)">;
}

let Features = "movrs,avx10.2", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vmovrsd512 : X86Builtin<"_Vector<16, int>(_Vector<16, int const *>)">;
}

let Features = "movrs,avx10.2", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vmovrsq128 : X86Builtin<"_Vector<2, long long int>(_Vector<2, long long int const *>)">;
}

let Features = "movrs,avx10.2", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vmovrsq256 : X86Builtin<"_Vector<4, long long int>(_Vector<4, long long int const *>)">;
}

let Features = "movrs,avx10.2", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vmovrsq512 : X86Builtin<"_Vector<8, long long int>(_Vector<8, long long int const *>)">;
}

let Features = "movrs,avx10.2", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def vmovrsw128 : X86Builtin<"_Vector<8, short>(_Vector<8, short const *>)">;
}

let Features = "movrs,avx10.2", Attributes = [NoThrow, RequiredVectorWidth<256>] in {
  def vmovrsw256 : X86Builtin<"_Vector<16, short>(_Vector<16, short const *>)">;
}

let Features = "movrs,avx10.2", Attributes = [NoThrow, RequiredVectorWidth<512>] in {
  def vmovrsw512 : X86Builtin<"_Vector<32, short>(_Vector<32, short const *>)">;
}
