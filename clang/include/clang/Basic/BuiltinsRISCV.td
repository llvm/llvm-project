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

//===----------------------------------------------------------------------===//
// Packed SIMD extension.
//===----------------------------------------------------------------------===//
let Features = "experimental-p,32bit" in {
def pslli_b_32  : RISCVBuiltin<"uint32_t(uint32_t, int32_t)">;
def pslli_h_32  : RISCVBuiltin<"uint32_t(uint32_t, int32_t)">;
def psslai_h_32 : RISCVBuiltin<"uint32_t(uint32_t, int32_t)">;
def sslai       : RISCVBuiltin<"uint32_t(uint32_t, int32_t)">;
def psll_bs_32  : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def psll_hs_32  : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def padd_bs_32  : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def padd_hs_32  : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def pusati_h_32 : RISCVBuiltin<"uint32_t(uint32_t, int32_t)">;
def usati_32    : RISCVBuiltin<"uint32_t(uint32_t, int32_t)">;
def psrai_b_32  : RISCVBuiltin<"uint32_t(uint32_t, int32_t)">;
def psrai_h_32  : RISCVBuiltin<"uint32_t(uint32_t, int32_t)">;
def psrari_h_32 : RISCVBuiltin<"uint32_t(uint32_t, int32_t)">;
def srari_32    : RISCVBuiltin<"int32_t(int32_t, int32_t)">;
def psati_h_32  : RISCVBuiltin<"uint32_t(uint32_t, int32_t)">;
def sati_32     : RISCVBuiltin<"int32_t(int32_t, int32_t)">;
def psrl_bs_32  : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def psrl_hs_32  : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def predsum_bs_32  : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def predsum_hs_32  : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def predsumu_bs_32 : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def predsumu_hs_32 : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def psra_bs_32     : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def psra_hs_32     : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def padd_b_32      : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def padd_h_32      : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def sadd           : RISCVBuiltin<"int32_t(int32_t, int32_t)">;
def psadd_b_32     : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def psadd_h_32     : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def aadd           : RISCVBuiltin<"int32_t(int32_t, int32_t)">;
def paadd_b_32     : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def paadd_h_32     : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def saddu          : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def psaddu_b_32    : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def psaddu_h_32    : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def aaddu          : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def paaddu_b_32    : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def paaddu_h_32    : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def psub_b_32      : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def psub_h_32      : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def ssub           : RISCVBuiltin<"int32_t(int32_t, int32_t)">;
def pssub_b_32     : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def pssub_h_32     : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def asub           : RISCVBuiltin<"int32_t(int32_t, int32_t)">;
def pasub_b_32     : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def pasub_h_32     : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def ssubu          : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def pssubu_b_32    : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def pssubu_h_32    : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def asubu          : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def pasubu_b_32    : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def pasubu_h_32    : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def pdif_b_32      : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def pdif_h_32      : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def pdifu_b_32     : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def pdifu_h_32     : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def pmul_h_b01_32  : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def pmulu_h_b01_32 : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def mul_h01        : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def mulu_h01       : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def slx_32         : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def psh1add_h_32   : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
def ssh1sadd       : RISCVBuiltin<"int32_t(int32_t, int32_t)">;
def pssh1sadd_h_32 : RISCVBuiltin<"uint32_t(uint32_t, uint32_t)">;
} // Features = "experimental-p,32bit"

let Features = "experimental-p,64bit" in {
def pslli_b_64  : RISCVBuiltin<"uint64_t(uint64_t, int64_t)">;
def pslli_h_64  : RISCVBuiltin<"uint64_t(uint64_t, int64_t)">;
def pslli_w     : RISCVBuiltin<"uint64_t(uint64_t, int64_t)">;
def psslai_h_64 : RISCVBuiltin<"uint64_t(uint64_t, int64_t)">;
def psslai_w    : RISCVBuiltin<"uint64_t(uint64_t, int64_t)">;
def psll_bs_64  : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def psll_hs_64  : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def psll_ws     : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def padd_bs_64  : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def padd_hs_64  : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def padd_ws     : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def pusati_h_64 : RISCVBuiltin<"uint64_t(uint64_t, int64_t)">;
def pusati_w    : RISCVBuiltin<"uint64_t(uint64_t, int64_t)">;
def usati_64    : RISCVBuiltin<"uint64_t(uint64_t, int64_t)">;
def psrai_b_64  : RISCVBuiltin<"uint64_t(uint64_t, int64_t)">;
def psrai_h_64  : RISCVBuiltin<"uint64_t(uint64_t, int64_t)">;
def psrai_w     : RISCVBuiltin<"uint64_t(uint64_t, int64_t)">;
def psrari_h_64 : RISCVBuiltin<"uint64_t(uint64_t, int64_t)">;
def psrari_w    : RISCVBuiltin<"uint64_t(uint64_t, int64_t)">;
def srari_64    : RISCVBuiltin<"int64_t(int64_t, int64_t)">;
def psati_h_64  : RISCVBuiltin<"uint64_t(uint64_t, int64_t)">;
def psati_w     : RISCVBuiltin<"uint64_t(uint64_t, int64_t)">;
def sati_64     : RISCVBuiltin<"int64_t(int64_t, int64_t)">;
def psrl_bs_64  : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def psrl_hs_64  : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def psrl_ws     : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def predsum_bs_64  : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def predsum_hs_64  : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def predsum_ws     : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def predsumu_bs_64 : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def predsumu_hs_64 : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def predsumu_ws    : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def psra_bs_64     : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def psra_hs_64     : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def psra_ws        : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def padd_b_64      : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def padd_h_64      : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def padd_w         : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def psadd_b_64     : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def psadd_h_64     : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def psadd_w        : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def paadd_b_64     : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def paadd_h_64     : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def paadd_w        : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def psaddu_b_64    : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def psaddu_h_64    : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def psaddu_w       : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def paaddu_b_64    : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def paaddu_h_64    : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def paaddu_w       : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def psub_b_64      : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def psub_h_64      : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def psub_w         : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def pssub_b_64     : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def pssub_h_64     : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def pssub_w        : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def pasub_b_64     : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def pasub_h_64     : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def pasub_w        : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def pssubu_b_64    : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def pssubu_h_64    : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def pssubu_w       : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def pasubu_b_64    : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def pasubu_h_64    : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def pasubu_w       : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def pdif_b_64      : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def pdif_h_64      : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def pdifu_b_64     : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def pdifu_h_64     : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def pmul_h_b01_64  : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def pmul_w_h01     : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def pmulu_h_b01_64 : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def pmulu_w_h01    : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def mul_w01        : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def mulu_w01       : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def slx_64         : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def psh1add_h_64   : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def psh1add_w      : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def pssh1sadd_h_64 : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def pssh1sadd_w    : RISCVBuiltin<"uint64_t(uint64_t, uint64_t)">;
def unzip8p        : RISCVBuiltin<"int64_t(int64_t, int64_t)">;
def unzip16p       : RISCVBuiltin<"int64_t(int64_t, int64_t)">;
def unzip8hp       : RISCVBuiltin<"int64_t(int64_t, int64_t)">;
def unzip16hp      : RISCVBuiltin<"int64_t(int64_t, int64_t)">;
def zip8p          : RISCVBuiltin<"int64_t(int64_t, int64_t)">;
def zip16p         : RISCVBuiltin<"int64_t(int64_t, int64_t)">;
def zip8hp         : RISCVBuiltin<"int64_t(int64_t, int64_t)">;
def zip16hp        : RISCVBuiltin<"int64_t(int64_t, int64_t)">;
} // Features = "experimental-p,64bit"
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
// XCV extensions.
//===----------------------------------------------------------------------===//
include "clang/Basic/BuiltinsRISCVXCV.td"

//===----------------------------------------------------------------------===//
// XAndes extensions.
//===----------------------------------------------------------------------===//
include "clang/Basic/BuiltinsRISCVXAndes.td"
