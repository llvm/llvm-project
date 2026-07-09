//==- BuiltinsRISCV.td - RISC-V Builtin function database ---*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the RISC-V-specific builtin function database.
//
//===----------------------------------------------------------------------===//

include "clang/Basic/BuiltinsBase.td"

class RISCVBuiltin<string prototype, string features = ""> : TargetBuiltin {
  let Spellings = ["__builtin_riscv_" # NAME];
  let Prototype = prototype;
  let Features = features;
}

let Attributes = [NoThrow, Const] in {
//===----------------------------------------------------------------------===//
// Zbb extension.
//===----------------------------------------------------------------------===//
def orc_b_32 : RISCVBuiltin<"unsigned int(unsigned int)", "zbb">;
def orc_b_64 : RISCVBuiltin<"uint64_t(uint64_t)", "zbb,64bit">;
def clz_32 : RISCVBuiltin<"unsigned int(unsigned int)", "zbb|xtheadbb">;
def clz_64 : RISCVBuiltin<"unsigned int(uint64_t)", "zbb|xtheadbb,64bit">;
def ctz_32 : RISCVBuiltin<"unsigned int(unsigned int)", "zbb">;
def ctz_64 : RISCVBuiltin<"unsigned int(uint64_t)", "zbb,64bit">;

//===----------------------------------------------------------------------===//
// Zbc or Zbkc extension.
//===----------------------------------------------------------------------===//
def clmul_32 : RISCVBuiltin<"unsigned int(unsigned int, unsigned int)", "zbc|zbkc">;
def clmul_64 : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)", "zbc|zbkc,64bit">;
def clmulh_32 : RISCVBuiltin<"unsigned int(unsigned int, unsigned int)", "zbc|zbkc,32bit">;
def clmulh_64 : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)", "zbc|zbkc,64bit">;
def clmulr_32 : RISCVBuiltin<"unsigned int(unsigned int, unsigned int)", "zbc,32bit">;
def clmulr_64 : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)", "zbc,64bit">;

//===----------------------------------------------------------------------===//
// Zbkx extension.
//===----------------------------------------------------------------------===//
let Features = "zbkx,32bit" in {
def xperm4_32 : RISCVBuiltin<"unsigned int(unsigned int, unsigned int)">;
def xperm8_32 : RISCVBuiltin<"unsigned int(unsigned int, unsigned int)">;
} // Features = "zbkx,32bit"

let Features = "zbkx,64bit" in {
def xperm4_64 : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def xperm8_64 : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
} // Features = "zbkx,64bit"

//===----------------------------------------------------------------------===//
// Zbkb extension.
//===----------------------------------------------------------------------===//
def brev8_32 : RISCVBuiltin<"unsigned int(unsigned int)", "zbkb">;
def brev8_64 : RISCVBuiltin<"uint64_t(uint64_t)", "zbkb,64bit">;
def zip_32 : RISCVBuiltin<"unsigned int(unsigned int)", "zbkb,32bit">;
def unzip_32 : RISCVBuiltin<"unsigned int(unsigned int)", "zbkb,32bit">;

//===----------------------------------------------------------------------===//
// Zknd extension.
//===----------------------------------------------------------------------===//
let Features = "zknd,32bit" in {
def aes32dsi : RISCVBuiltin<"unsigned int(unsigned int, unsigned int, _Constant unsigned int)">;
def aes32dsmi : RISCVBuiltin<"unsigned int(unsigned int, unsigned int, _Constant unsigned int)">;
} // Features = "zknd,32bit"

let Features = "zknd,64bit" in {
def aes64ds : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def aes64dsm : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def aes64im : RISCVBuiltin<"uint64_t(uint64_t)">;
} // Features = "zknd,64bit"

//===----------------------------------------------------------------------===//
// Zknd & Zkne extension.
//===----------------------------------------------------------------------===//
let Features = "zknd|zkne,64bit" in {
def aes64ks1i : RISCVBuiltin<"uint64_t(uint64_t, _Constant unsigned int)">;
def aes64ks2 : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
} // Features = "zknd|zkne,64bit"

//===----------------------------------------------------------------------===//
// Zkne extension.
//===----------------------------------------------------------------------===//
let Features = "zkne,32bit" in {
def aes32esi : RISCVBuiltin<"unsigned int(unsigned int, unsigned int, _Constant unsigned int)">;
def aes32esmi : RISCVBuiltin<"unsigned int(unsigned int, unsigned int, _Constant unsigned int)">;
} // Features = "zkne,32bit"

let Features = "zkne,64bit" in {
def aes64es : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def aes64esm : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
} // Features = "zkne,64bit"

//===----------------------------------------------------------------------===//
// Zknh extension.
//===----------------------------------------------------------------------===//
let Features = "zknh" in {
def sha256sig0 : RISCVBuiltin<"unsigned int(unsigned int)">;
def sha256sig1 : RISCVBuiltin<"unsigned int(unsigned int)">;
def sha256sum0 : RISCVBuiltin<"unsigned int(unsigned int)">;
def sha256sum1 : RISCVBuiltin<"unsigned int(unsigned int)">;
} // Features = "zknh"

let Features = "zknh,32bit" in {
def sha512sig0h : RISCVBuiltin<"unsigned int(unsigned int, unsigned int)">;
def sha512sig0l : RISCVBuiltin<"unsigned int(unsigned int, unsigned int)">;
def sha512sig1h : RISCVBuiltin<"unsigned int(unsigned int, unsigned int)">;
def sha512sig1l : RISCVBuiltin<"unsigned int(unsigned int, unsigned int)">;
def sha512sum0r : RISCVBuiltin<"unsigned int(unsigned int, unsigned int)">;
def sha512sum1r : RISCVBuiltin<"unsigned int(unsigned int, unsigned int)">;
} // Features = "zknh,32bit"

let Features = "zknh,64bit" in {
def sha512sig0 : RISCVBuiltin<"uint64_t(uint64_t)">;
def sha512sig1 : RISCVBuiltin<"uint64_t(uint64_t)">;
def sha512sum0 : RISCVBuiltin<"uint64_t(uint64_t)">;
def sha512sum1 : RISCVBuiltin<"uint64_t(uint64_t)">;
} // Features = "zknh,64bit"

//===----------------------------------------------------------------------===//
// Zksed extension.
//===----------------------------------------------------------------------===//
let Features = "zksed" in {
def sm4ed : RISCVBuiltin<"unsigned int(unsigned int, unsigned int, _Constant unsigned int )">;
def sm4ks : RISCVBuiltin<"unsigned int(unsigned int, unsigned int, _Constant unsigned int)">;
} // Features = "zksed"

//===----------------------------------------------------------------------===//
// Zksh extension.
//===----------------------------------------------------------------------===//
let Features = "zksh" in {
def sm3p0 : RISCVBuiltin<"unsigned int(unsigned int)">;
def sm3p1 : RISCVBuiltin<"unsigned int(unsigned int)">;
} // Features = "zksh"

} // Attributes = [Const, NoThrow]

//===----------------------------------------------------------------------===//
// Zihintntl extension.
//===----------------------------------------------------------------------===//
let Features = "zihintntl", Attributes = [CustomTypeChecking] in {
def ntl_load : RISCVBuiltin<"void(...)">;
def ntl_store : RISCVBuiltin<"void(...)">;
} // Features = "zihintntl", Attributes = [CustomTypeChecking]

//===----------------------------------------------------------------------===//
// Zihintpause extension.
//===----------------------------------------------------------------------===//
let Features = "zihintpause", Attributes = [NoThrow] in
def pause : RISCVBuiltin<"void()">;

//===----------------------------------------------------------------------===//
// P extension.
//===----------------------------------------------------------------------===//
let Features = "experimental-p", Attributes = [NoThrow, Const] in {
// Packed Averaging Addition and Subtraction (32-bit)
def paadd_i8x4 : RISCVBuiltin<"_Vector<4, signed char>(_Vector<4, signed char>, _Vector<4, signed char>)">;
def paadd_i16x2 : RISCVBuiltin<"_Vector<2, short>(_Vector<2, short>, _Vector<2, short>)">;
def paaddu_u8x4 : RISCVBuiltin<"_Vector<4, unsigned char>(_Vector<4, unsigned char>, _Vector<4, unsigned char>)">;
def paaddu_u16x2 : RISCVBuiltin<"_Vector<2, unsigned short>(_Vector<2, unsigned short>, _Vector<2, unsigned short>)">;
def pasub_i8x4 : RISCVBuiltin<"_Vector<4, signed char>(_Vector<4, signed char>, _Vector<4, signed char>)">;
def pasub_i16x2 : RISCVBuiltin<"_Vector<2, short>(_Vector<2, short>, _Vector<2, short>)">;
def pasubu_u8x4 : RISCVBuiltin<"_Vector<4, unsigned char>(_Vector<4, unsigned char>, _Vector<4, unsigned char>)">;
def pasubu_u16x2 : RISCVBuiltin<"_Vector<2, unsigned short>(_Vector<2, unsigned short>, _Vector<2, unsigned short>)">;

// Packed Averaging Addition and Subtraction (64-bit)
def paadd_i8x8 : RISCVBuiltin<"_Vector<8, signed char>(_Vector<8, signed char>, _Vector<8, signed char>)">;
def paadd_i16x4 : RISCVBuiltin<"_Vector<4, short>(_Vector<4, short>, _Vector<4, short>)">;
def paadd_i32x2 : RISCVBuiltin<"_Vector<2, int>(_Vector<2, int>, _Vector<2, int>)">;
def paaddu_u8x8 : RISCVBuiltin<"_Vector<8, unsigned char>(_Vector<8, unsigned char>, _Vector<8, unsigned char>)">;
def paaddu_u16x4 : RISCVBuiltin<"_Vector<4, unsigned short>(_Vector<4, unsigned short>, _Vector<4, unsigned short>)">;
def paaddu_u32x2 : RISCVBuiltin<"_Vector<2, unsigned int>(_Vector<2, unsigned int>, _Vector<2, unsigned int>)">;
def pasub_i8x8 : RISCVBuiltin<"_Vector<8, signed char>(_Vector<8, signed char>, _Vector<8, signed char>)">;
def pasub_i16x4 : RISCVBuiltin<"_Vector<4, short>(_Vector<4, short>, _Vector<4, short>)">;
def pasub_i32x2 : RISCVBuiltin<"_Vector<2, int>(_Vector<2, int>, _Vector<2, int>)">;
def pasubu_u8x8 : RISCVBuiltin<"_Vector<8, unsigned char>(_Vector<8, unsigned char>, _Vector<8, unsigned char>)">;
def pasubu_u16x4 : RISCVBuiltin<"_Vector<4, unsigned short>(_Vector<4, unsigned short>, _Vector<4, unsigned short>)">;
def pasubu_u32x2 : RISCVBuiltin<"_Vector<2, unsigned int>(_Vector<2, unsigned int>, _Vector<2, unsigned int>)">;

// Packed Exchanged Addition and Subtraction (32-bit)
def pas_x_i16x2 : RISCVBuiltin<"_Vector<2, short>(_Vector<2, short>, _Vector<2, short>)">;
def psa_x_i16x2 : RISCVBuiltin<"_Vector<2, short>(_Vector<2, short>, _Vector<2, short>)">;
def psas_x_i16x2 : RISCVBuiltin<"_Vector<2, short>(_Vector<2, short>, _Vector<2, short>)">;
def pssa_x_i16x2 : RISCVBuiltin<"_Vector<2, short>(_Vector<2, short>, _Vector<2, short>)">;
def paas_x_i16x2 : RISCVBuiltin<"_Vector<2, short>(_Vector<2, short>, _Vector<2, short>)">;
def pasa_x_i16x2 : RISCVBuiltin<"_Vector<2, short>(_Vector<2, short>, _Vector<2, short>)">;

// Packed Exchanged Addition and Subtraction (64-bit)
def pas_x_i16x4 : RISCVBuiltin<"_Vector<4, short>(_Vector<4, short>, _Vector<4, short>)">;
def pas_x_i32x2 : RISCVBuiltin<"_Vector<2, int>(_Vector<2, int>, _Vector<2, int>)">;
def psa_x_i16x4 : RISCVBuiltin<"_Vector<4, short>(_Vector<4, short>, _Vector<4, short>)">;
def psa_x_i32x2 : RISCVBuiltin<"_Vector<2, int>(_Vector<2, int>, _Vector<2, int>)">;
def psas_x_i16x4 : RISCVBuiltin<"_Vector<4, short>(_Vector<4, short>, _Vector<4, short>)">;
def psas_x_i32x2 : RISCVBuiltin<"_Vector<2, int>(_Vector<2, int>, _Vector<2, int>)">;
def pssa_x_i16x4 : RISCVBuiltin<"_Vector<4, short>(_Vector<4, short>, _Vector<4, short>)">;
def pssa_x_i32x2 : RISCVBuiltin<"_Vector<2, int>(_Vector<2, int>, _Vector<2, int>)">;
def paas_x_i16x4 : RISCVBuiltin<"_Vector<4, short>(_Vector<4, short>, _Vector<4, short>)">;
def paas_x_i32x2 : RISCVBuiltin<"_Vector<2, int>(_Vector<2, int>, _Vector<2, int>)">;
def pasa_x_i16x4 : RISCVBuiltin<"_Vector<4, short>(_Vector<4, short>, _Vector<4, short>)">;
def pasa_x_i32x2 : RISCVBuiltin<"_Vector<2, int>(_Vector<2, int>, _Vector<2, int>)">;

// Packed Absolute Value and Absolute Difference (32-bit)
def pabd_i8x4 : RISCVBuiltin<"_Vector<4, unsigned char>(_Vector<4, signed char>, _Vector<4, signed char>)">;
def pabd_i16x2 : RISCVBuiltin<"_Vector<2, unsigned short>(_Vector<2, short>, _Vector<2, short>)">;
def pabdu_u8x4 : RISCVBuiltin<"_Vector<4, unsigned char>(_Vector<4, unsigned char>, _Vector<4, unsigned char>)">;
def pabdu_u16x2 : RISCVBuiltin<"_Vector<2, unsigned short>(_Vector<2, unsigned short>, _Vector<2, unsigned short>)">;

// Packed Absolute Value and Absolute Difference (64-bit)
def pabd_i8x8 : RISCVBuiltin<"_Vector<8, unsigned char>(_Vector<8, signed char>, _Vector<8, signed char>)">;
def pabd_i16x4 : RISCVBuiltin<"_Vector<4, unsigned short>(_Vector<4, short>, _Vector<4, short>)">;
def pabdu_u8x8 : RISCVBuiltin<"_Vector<8, unsigned char>(_Vector<8, unsigned char>, _Vector<8, unsigned char>)">;
def pabdu_u16x4 : RISCVBuiltin<"_Vector<4, unsigned short>(_Vector<4, unsigned short>, _Vector<4, unsigned short>)">;

// Packed Reduction Sum (32-bit)
def predsum_i8x4_i32 : RISCVBuiltin<"int(_Vector<4, signed char>, int)">;
def predsumu_u8x4_u32 : RISCVBuiltin<"unsigned int(_Vector<4, unsigned char>, unsigned int)">;
def predsum_i16x2_i32 : RISCVBuiltin<"int(_Vector<2, short>, int)">;
def predsumu_u16x2_u32 : RISCVBuiltin<"unsigned int(_Vector<2, unsigned short>, unsigned int)">;

// Packed Reduction Sum (64-bit)
def predsum_i8x8_i32 : RISCVBuiltin<"int(_Vector<8, signed char>, int)">;
def predsumu_u8x8_u32 : RISCVBuiltin<"unsigned int(_Vector<8, unsigned char>, unsigned int)">;
def predsum_i16x4_i32 : RISCVBuiltin<"int(_Vector<4, short>, int)">;
def predsumu_u16x4_u32 : RISCVBuiltin<"unsigned int(_Vector<4, unsigned short>, unsigned int)">;
def predsum_i8x8_i64 : RISCVBuiltin<"int64_t(_Vector<8, signed char>, int64_t)">;
def predsumu_u8x8_u64 : RISCVBuiltin<"uint64_t(_Vector<8, unsigned char>, uint64_t)">;
def predsum_i16x4_i64 : RISCVBuiltin<"int64_t(_Vector<4, short>, int64_t)">;
def predsumu_u16x4_u64 : RISCVBuiltin<"uint64_t(_Vector<4, unsigned short>, uint64_t)">;
def predsum_i32x2_i64 : RISCVBuiltin<"int64_t(_Vector<2, int>, int64_t)">;
def predsumu_u32x2_u64 : RISCVBuiltin<"uint64_t(_Vector<2, unsigned int>, uint64_t)">;

// Packed Merge (32-bit)
def pmerge_u8x4 : RISCVBuiltin<"_Vector<4, unsigned char>(_Vector<4, unsigned char>, _Vector<4, unsigned char>, _Vector<4, unsigned char>)">;
def pmerge_i8x4 : RISCVBuiltin<"_Vector<4, signed char>(_Vector<4, signed char>, _Vector<4, signed char>, _Vector<4, unsigned char>)">;
def pmerge_u16x2 : RISCVBuiltin<"_Vector<2, unsigned short>(_Vector<2, unsigned short>, _Vector<2, unsigned short>, _Vector<2, unsigned short>)">;
def pmerge_i16x2 : RISCVBuiltin<"_Vector<2, short>(_Vector<2, short>, _Vector<2, short>, _Vector<2, unsigned short>)">;

// Packed Merge (64-bit)
def pmerge_u8x8 : RISCVBuiltin<"_Vector<8, unsigned char>(_Vector<8, unsigned char>, _Vector<8, unsigned char>, _Vector<8, unsigned char>)">;
def pmerge_i8x8 : RISCVBuiltin<"_Vector<8, signed char>(_Vector<8, signed char>, _Vector<8, signed char>, _Vector<8, unsigned char>)">;
def pmerge_u16x4 : RISCVBuiltin<"_Vector<4, unsigned short>(_Vector<4, unsigned short>, _Vector<4, unsigned short>, _Vector<4, unsigned short>)">;
def pmerge_i16x4 : RISCVBuiltin<"_Vector<4, short>(_Vector<4, short>, _Vector<4, short>, _Vector<4, unsigned short>)">;
def pmerge_u32x2 : RISCVBuiltin<"_Vector<2, unsigned int>(_Vector<2, unsigned int>, _Vector<2, unsigned int>, _Vector<2, unsigned int>)">;
def pmerge_i32x2 : RISCVBuiltin<"_Vector<2, int>(_Vector<2, int>, _Vector<2, int>, _Vector<2, unsigned int>)">;

// Packed Absolute Difference Sum (32-bit)
def pabdsumu_u8x4_u32 : RISCVBuiltin<"unsigned int(_Vector<4, unsigned char>, _Vector<4, unsigned char>)">;
def pabdsumau_u8x4_u32 : RISCVBuiltin<"unsigned int(unsigned int, _Vector<4, unsigned char>, _Vector<4, unsigned char>)">;

// Packed Absolute Difference Sum (64-bit)
def pabdsumu_u8x8_u32 : RISCVBuiltin<"unsigned int(_Vector<8, unsigned char>, _Vector<8, unsigned char>)">;
def pabdsumu_u8x8_u64 : RISCVBuiltin<"uint64_t(_Vector<8, unsigned char>, _Vector<8, unsigned char>)">;
def pabdsumau_u8x8_u32 : RISCVBuiltin<"unsigned int(unsigned int, _Vector<8, unsigned char>, _Vector<8, unsigned char>)">;
def pabdsumau_u8x8_u64 : RISCVBuiltin<"uint64_t(uint64_t, _Vector<8, unsigned char>, _Vector<8, unsigned char>)">;

// Packed Saturating Absolute Value (32-bit)
def psabs_i8x4 : RISCVBuiltin<"_Vector<4, signed char>(_Vector<4, signed char>)">;
def psabs_i16x2 : RISCVBuiltin<"_Vector<2, short>(_Vector<2, short>)">;

// Packed Saturating Absolute Value (64-bit)
def psabs_i8x8 : RISCVBuiltin<"_Vector<8, signed char>(_Vector<8, signed char>)">;
def psabs_i16x4 : RISCVBuiltin<"_Vector<4, short>(_Vector<4, short>)">;

} // Features = "experimental-p"

//===----------------------------------------------------------------------===//
// XCV extensions.
//===----------------------------------------------------------------------===//
include "clang/Basic/BuiltinsRISCVXCV.td"

//===----------------------------------------------------------------------===//
// XAndes extensions.
//===----------------------------------------------------------------------===//
include "clang/Basic/BuiltinsRISCVXAndes.td"

//===----------------------------------------------------------------------===//
// MIPS extensions.
//===----------------------------------------------------------------------===//
include "clang/Basic/BuiltinsRISCVXMIPS.td"
