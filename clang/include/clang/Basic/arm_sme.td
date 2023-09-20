//===--- arm_sme.td - ARM SME compiler interface ------------------------===//
//
//  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the TableGen definitions from which the ARM SME header
//  file will be generated.  See:
//
//      https://developer.arm.com/architectures/system-architectures/software-standards/acle
//
//===----------------------------------------------------------------------===//

include "arm_sve_sme_incl.td"

////////////////////////////////////////////////////////////////////////////////
// Loads

multiclass ZALoad<string n_suffix, string t, string i_prefix, list<ImmCheck> ch> {
  let TargetGuard = "sme" in {
    def NAME # _H : MInst<"svld1_hor_" # n_suffix, "vimPQ", t,
                          [IsLoad, IsOverloadNone, IsStreaming, IsSharedZA],
                          MemEltTyDefault, i_prefix # "_horiz", ch>;

    def NAME # _H_VNUM : MInst<"svld1_hor_vnum_" # n_suffix, "vimPQl", t,
                               [IsLoad, IsOverloadNone, IsStreaming, IsSharedZA],
                               MemEltTyDefault, i_prefix # "_horiz", ch>;

    def NAME # _V : MInst<"svld1_ver_" # n_suffix, "vimPQ", t,
                          [IsLoad, IsOverloadNone, IsStreaming, IsSharedZA],
                          MemEltTyDefault, i_prefix # "_vert", ch>;

    def NAME # _V_VNUM : MInst<"svld1_ver_vnum_" # n_suffix, "vimPQl", t,
                               [IsLoad, IsOverloadNone, IsStreaming, IsSharedZA],
                               MemEltTyDefault, i_prefix # "_vert", ch>;
  }
}

defm SVLD1_ZA8 : ZALoad<"za8", "c", "aarch64_sme_ld1b", [ImmCheck<0, ImmCheck0_0>]>;
defm SVLD1_ZA16 : ZALoad<"za16", "s", "aarch64_sme_ld1h", [ImmCheck<0, ImmCheck0_1>]>;
defm SVLD1_ZA32 : ZALoad<"za32", "i", "aarch64_sme_ld1w", [ImmCheck<0, ImmCheck0_3>]>;
defm SVLD1_ZA64 : ZALoad<"za64", "l", "aarch64_sme_ld1d", [ImmCheck<0, ImmCheck0_7>]>;
defm SVLD1_ZA128 : ZALoad<"za128", "q", "aarch64_sme_ld1q", [ImmCheck<0, ImmCheck0_15>]>;

def SVLDR_VNUM_ZA : MInst<"svldr_vnum_za", "vmQi", "",
                          [IsOverloadNone, IsStreamingCompatible, IsSharedZA],
                          MemEltTyDefault, "aarch64_sme_ldr",
                          [ImmCheck<2, ImmCheck0_15>]>;

def SVLDR_ZA : MInst<"svldr_za", "vmQ", "",
                          [IsOverloadNone, IsStreamingCompatible, IsSharedZA],
                          MemEltTyDefault, "aarch64_sme_ldr", []>;

////////////////////////////////////////////////////////////////////////////////
// Stores

multiclass ZAStore<string n_suffix, string t, string i_prefix, list<ImmCheck> ch> {
  let TargetGuard = "sme" in {
    def NAME # _H : MInst<"svst1_hor_" # n_suffix, "vimP%", t,
                          [IsStore, IsOverloadNone, IsStreaming, IsSharedZA, IsPreservesZA],
                          MemEltTyDefault, i_prefix # "_horiz", ch>;

    def NAME # _H_VNUM : MInst<"svst1_hor_vnum_" # n_suffix, "vimP%l", t,
                               [IsStore, IsOverloadNone, IsStreaming, IsSharedZA, IsPreservesZA],
                               MemEltTyDefault, i_prefix # "_horiz", ch>;

    def NAME # _V : MInst<"svst1_ver_" # n_suffix, "vimP%", t,
                          [IsStore, IsOverloadNone, IsStreaming, IsSharedZA, IsPreservesZA],
                          MemEltTyDefault, i_prefix # "_vert", ch>;

    def NAME # _V_VNUM : MInst<"svst1_ver_vnum_" # n_suffix, "vimP%l", t,
                               [IsStore, IsOverloadNone, IsStreaming, IsSharedZA, IsPreservesZA],
                               MemEltTyDefault, i_prefix # "_vert", ch>;
  }
}

defm SVST1_ZA8 : ZAStore<"za8", "c", "aarch64_sme_st1b", [ImmCheck<0, ImmCheck0_0>]>;
defm SVST1_ZA16 : ZAStore<"za16", "s", "aarch64_sme_st1h", [ImmCheck<0, ImmCheck0_1>]>;
defm SVST1_ZA32 : ZAStore<"za32", "i", "aarch64_sme_st1w", [ImmCheck<0, ImmCheck0_3>]>;
defm SVST1_ZA64 : ZAStore<"za64", "l", "aarch64_sme_st1d", [ImmCheck<0, ImmCheck0_7>]>;
defm SVST1_ZA128 : ZAStore<"za128", "q", "aarch64_sme_st1q", [ImmCheck<0, ImmCheck0_15>]>;

def SVSTR_VNUM_ZA : MInst<"svstr_vnum_za", "vm%i", "",
                          [IsOverloadNone, IsStreamingCompatible, IsSharedZA, IsPreservesZA],
                          MemEltTyDefault, "aarch64_sme_str",
                          [ImmCheck<2, ImmCheck0_15>]>;

def SVSTR_ZA : MInst<"svstr_za", "vm%", "",
                      [IsOverloadNone, IsStreamingCompatible, IsSharedZA, IsPreservesZA],
                      MemEltTyDefault, "aarch64_sme_str", []>;

////////////////////////////////////////////////////////////////////////////////
// Read horizontal/vertical ZA slices

multiclass ZARead<string n_suffix, string t, string i_prefix, list<ImmCheck> ch> {
  let TargetGuard = "sme" in {
    def NAME # _H : SInst<"svread_hor_" # n_suffix # "[_{d}]", "ddPim", t,
                          MergeOp1, i_prefix # "_horiz",
                          [IsReadZA, IsStreaming, IsSharedZA, IsPreservesZA], ch>;

    def NAME # _V : SInst<"svread_ver_" # n_suffix # "[_{d}]", "ddPim", t,
                          MergeOp1, i_prefix # "_vert",
                          [IsReadZA, IsStreaming, IsSharedZA, IsPreservesZA], ch>;
  }
}

defm SVREAD_ZA8 : ZARead<"za8", "cUc", "aarch64_sme_read", [ImmCheck<2, ImmCheck0_0>]>;
defm SVREAD_ZA16 : ZARead<"za16", "sUshb", "aarch64_sme_read", [ImmCheck<2, ImmCheck0_1>]>;
defm SVREAD_ZA32 : ZARead<"za32", "iUif", "aarch64_sme_read", [ImmCheck<2, ImmCheck0_3>]>;
defm SVREAD_ZA64 : ZARead<"za64", "lUld", "aarch64_sme_read", [ImmCheck<2, ImmCheck0_7>]>;
defm SVREAD_ZA128 : ZARead<"za128", "csilUcUsUiUlhbfd", "aarch64_sme_readq", [ImmCheck<2, ImmCheck0_15>]>;

////////////////////////////////////////////////////////////////////////////////
// Write horizontal/vertical ZA slices

multiclass ZAWrite<string n_suffix, string t, string i_prefix, list<ImmCheck> ch> {
  let TargetGuard = "sme" in {
    def NAME # _H : SInst<"svwrite_hor_" # n_suffix # "[_{d}]", "vimPd", t,
                          MergeOp1, i_prefix # "_horiz",
                          [IsWriteZA, IsStreaming, IsSharedZA], ch>;

    def NAME # _V : SInst<"svwrite_ver_" # n_suffix # "[_{d}]", "vimPd", t,
                          MergeOp1, i_prefix # "_vert",
                          [IsWriteZA, IsStreaming, IsSharedZA], ch>;
  }
}

defm SVWRITE_ZA8 : ZAWrite<"za8", "cUc", "aarch64_sme_write", [ImmCheck<0, ImmCheck0_0>]>;
defm SVWRITE_ZA16 : ZAWrite<"za16", "sUshb", "aarch64_sme_write", [ImmCheck<0, ImmCheck0_1>]>;
defm SVWRITE_ZA32 : ZAWrite<"za32", "iUif", "aarch64_sme_write", [ImmCheck<0, ImmCheck0_3>]>;
defm SVWRITE_ZA64 : ZAWrite<"za64", "lUld", "aarch64_sme_write", [ImmCheck<0, ImmCheck0_7>]>;
defm SVWRITE_ZA128 : ZAWrite<"za128", "csilUcUsUiUlhbfd", "aarch64_sme_writeq", [ImmCheck<0, ImmCheck0_15>]>;

////////////////////////////////////////////////////////////////////////////////
// SME - Zero

let TargetGuard = "sme" in {
  def SVZERO_MASK_ZA : SInst<"svzero_mask_za", "vi", "", MergeNone, "aarch64_sme_zero",
                             [IsOverloadNone, IsStreamingCompatible, IsSharedZA],
                             [ImmCheck<0, ImmCheck0_255>]>;
  def SVZERO_ZA      : SInst<"svzero_za", "v", "", MergeNone, "aarch64_sme_zero",
                             [IsOverloadNone, IsStreamingCompatible, IsSharedZA]>;
}

////////////////////////////////////////////////////////////////////////////////
// SME - Counting elements in a streaming vector

multiclass ZACount<string n_suffix> {
  let TargetGuard = "sme" in {
    def NAME : SInst<"sv" # n_suffix, "nv", "", MergeNone,
                      "aarch64_sme_" # n_suffix,
                      [IsOverloadNone, IsStreamingCompatible, IsPreservesZA]>;
  }
}

defm SVCNTSB : ZACount<"cntsb">;
defm SVCNTSH : ZACount<"cntsh">;
defm SVCNTSW : ZACount<"cntsw">;
defm SVCNTSD : ZACount<"cntsd">;

////////////////////////////////////////////////////////////////////////////////
// SME - ADDHA/ADDVA

multiclass ZAAdd<string n_suffix> {
  let TargetGuard = "sme" in {
    def NAME # _ZA32: SInst<"sv" # n_suffix # "_za32[_{d}]", "viPPd", "iUi", MergeOp1,
                      "aarch64_sme_" # n_suffix, [IsStreaming, IsSharedZA],
                      [ImmCheck<0, ImmCheck0_3>]>;
  }

  let TargetGuard = "sme-i16i64" in {
    def NAME # _ZA64: SInst<"sv" # n_suffix # "_za64[_{d}]", "viPPd", "lUl", MergeOp1,
                     "aarch64_sme_" # n_suffix, [IsStreaming, IsSharedZA],
                     [ImmCheck<0, ImmCheck0_7>]>;
  }
}

defm SVADDHA : ZAAdd<"addha">;
defm SVADDVA : ZAAdd<"addva">;

////////////////////////////////////////////////////////////////////////////////
// SME - SMOPA, SMOPS, UMOPA, UMOPS

multiclass ZAIntOuterProd<string n_suffix1, string n_suffix2> {
  let TargetGuard = "sme" in {
    def NAME # _ZA32_B: SInst<"sv" # n_suffix2 # "_za32[_{d}]",
                              "viPPdd", !cond(!eq(n_suffix1, "s") : "", true: "U") # "c",
                              MergeOp1, "aarch64_sme_" # n_suffix1 # n_suffix2 # "_wide",
                              [IsStreaming, IsSharedZA],
                              [ImmCheck<0, ImmCheck0_3>]>;
  }

  let TargetGuard = "sme-i16i64" in {
    def NAME # _ZA64_H: SInst<"sv" # n_suffix2 # "_za64[_{d}]",
                              "viPPdd", !cond(!eq(n_suffix1, "s") : "", true: "U") # "s",
                              MergeOp1, "aarch64_sme_" # n_suffix1 # n_suffix2 # "_wide",
                              [IsStreaming, IsSharedZA],
                              [ImmCheck<0, ImmCheck0_7>]>;
  }
}

defm SVSMOPA : ZAIntOuterProd<"s", "mopa">;
defm SVSMOPS : ZAIntOuterProd<"s", "mops">;
defm SVUMOPA : ZAIntOuterProd<"u", "mopa">;
defm SVUMOPS : ZAIntOuterProd<"u", "mops">;

////////////////////////////////////////////////////////////////////////////////
// SME - SUMOPA, SUMOPS, USMOPA, USMOPS

multiclass ZAIntOuterProdMixedSigns<string n_suffix1, string n_suffix2> {
  let TargetGuard = "sme" in {
    def NAME # _ZA32_B: SInst<"sv" # n_suffix1 # n_suffix2 # "_za32[_{d}]",
                              "viPPd" # !cond(!eq(n_suffix1, "su") : "u", true: "x"),
                              !cond(!eq(n_suffix1, "su") : "", true: "U") # "c",
                              MergeOp1, "aarch64_sme_" # n_suffix1 # n_suffix2 # "_wide",
                              [IsStreaming, IsSharedZA],
                              [ImmCheck<0, ImmCheck0_3>]>;
  }

  let TargetGuard = "sme-i16i64" in {
    def NAME # _ZA64_H: SInst<"sv" # n_suffix1 # n_suffix2 # "_za64[_{d}]",
                              "viPPd" # !cond(!eq(n_suffix1, "su") : "u", true: "x"),
                              !cond(!eq(n_suffix1, "su") : "", true: "U") # "s",
                              MergeOp1, "aarch64_sme_" # n_suffix1 # n_suffix2 # "_wide",
                              [IsStreaming, IsSharedZA],
                              [ImmCheck<0, ImmCheck0_7>]>;
  }
}

defm SVSUMOPA : ZAIntOuterProdMixedSigns<"su", "mopa">;
defm SVSUMOPS : ZAIntOuterProdMixedSigns<"su", "mops">;
defm SVUSMOPA : ZAIntOuterProdMixedSigns<"us", "mopa">;
defm SVUSMOPS : ZAIntOuterProdMixedSigns<"us", "mops">;

////////////////////////////////////////////////////////////////////////////////
// SME - FMOPA, FMOPS

multiclass ZAFPOuterProd<string n_suffix> {
  let TargetGuard = "sme" in {
    def NAME # _ZA32_B: SInst<"sv" # n_suffix # "_za32[_{d}]", "viPPdd", "h",
                              MergeOp1, "aarch64_sme_" # n_suffix # "_wide",
                              [IsStreaming, IsSharedZA],
                              [ImmCheck<0, ImmCheck0_3>]>;

    def NAME # _ZA32_H: SInst<"sv" # n_suffix # "_za32[_{d}]", "viPPdd", "b",
                              MergeOp1, "aarch64_sme_" # n_suffix # "_wide",
                              [IsStreaming, IsSharedZA],
                              [ImmCheck<0, ImmCheck0_3>]>;

    def NAME # _ZA32_S: SInst<"sv" # n_suffix # "_za32[_{d}]", "viPPdd", "f",
                              MergeOp1, "aarch64_sme_" # n_suffix,
                              [IsStreaming, IsSharedZA],
                              [ImmCheck<0, ImmCheck0_3>]>;
  }

  let TargetGuard = "sme-f64f64" in {
    def NAME # _ZA64_D: SInst<"sv" # n_suffix # "_za64[_{d}]", "viPPdd", "d",
                              MergeOp1, "aarch64_sme_" # n_suffix,
                              [IsStreaming, IsSharedZA],
                              [ImmCheck<0, ImmCheck0_7>]>;
  }
}

defm SVMOPA : ZAFPOuterProd<"mopa">;
defm SVMOPS : ZAFPOuterProd<"mops">;
