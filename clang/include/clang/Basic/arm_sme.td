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

def SVLDR_VNUM_ZA : MInst<"svldr_vnum_za", "vmQl", "",
                          [IsOverloadNone, IsStreamingCompatible, IsSharedZA],
                          MemEltTyDefault, "aarch64_sme_ldr">;

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

def SVSTR_VNUM_ZA : MInst<"svstr_vnum_za", "vm%l", "",
                          [IsOverloadNone, IsStreamingCompatible, IsSharedZA, IsPreservesZA],
                          MemEltTyDefault, "aarch64_sme_str">;

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

////////////////////////////////////////////////////////////////////////////////
// SME2 - ADD, SUB

multiclass ZAAddSub<string n_suffix> {
  let TargetGuard = "sme2" in {
    def NAME # _WRITE_SINGLE_ZA32_VG1X2_I32 : Inst<"sv" # n_suffix # "_write[_single]_za32[_{d}]_vg1x2", "vm2d", "iUi", MergeNone, "aarch64_sme_" # n_suffix # "_write_single_za_vg1x2", [IsStreaming, IsSharedZA], []>;
    def NAME # _WRITE_SINGLE_ZA32_VG1X4_I32 : Inst<"sv" # n_suffix # "_write[_single]_za32[_{d}]_vg1x4", "vm4d", "iUi", MergeNone, "aarch64_sme_" # n_suffix # "_write_single_za_vg1x4", [IsStreaming, IsSharedZA], []>;

    def NAME # _WRITE_ZA32_VG1X2_I32 : Inst<"sv" # n_suffix # "_write_za32[_{d}]_vg1x2", "vm22", "iUi", MergeNone, "aarch64_sme_" # n_suffix # "_write_za_vg1x2", [IsStreaming, IsSharedZA], []>;
    def NAME # _WRITE_ZA32_VG1X4_I32 : Inst<"sv" # n_suffix # "_write_za32[_{d}]_vg1x4", "vm44", "iUi", MergeNone, "aarch64_sme_" # n_suffix # "_write_za_vg1x4", [IsStreaming, IsSharedZA], []>;

    def NAME # _ZA32_VG1x2_I32 : Inst<"sv" # n_suffix # "_za32[_{d}]_vg1x2", "vm2", "iUif", MergeNone, "aarch64_sme_" # n_suffix # "_za32_vg1x2", [IsStreaming, IsSharedZA], []>;
    def NAME # _ZA32_VG1X4_I32 : Inst<"sv" # n_suffix # "_za32[_{d}]_vg1x4", "vm4", "iUif", MergeNone, "aarch64_sme_" # n_suffix # "_za32_vg1x4", [IsStreaming, IsSharedZA], []>;

    let TargetGuard = "sme-i16i64" in {
      def NAME # _WRITE_SINGLE_ZA64_VG1X2_I64 : Inst<"sv" # n_suffix # "_write[_single]_za64[_{d}]_vg1x2", "vm2d", "lUl", MergeNone, "aarch64_sme_" # n_suffix # "_write_single_za_vg1x2", [IsStreaming, IsSharedZA], []>;
      def NAME # _WRITE_SINGLE_ZA64_VG1X4_I64 : Inst<"sv" # n_suffix # "_write[_single]_za64[_{d}]_vg1x4", "vm4d", "lUl", MergeNone, "aarch64_sme_" # n_suffix # "_write_single_za_vg1x4", [IsStreaming, IsSharedZA], []>;

      def NAME # _WRITE_ZA64_VG1x2_I64 : Inst<"sv" # n_suffix # "_write_za64[_{d}]_vg1x2", "vm22", "lUl", MergeNone, "aarch64_sme_" # n_suffix # "_write_za_vg1x2", [IsStreaming, IsSharedZA], []>;
      def NAME # _WRITE_ZA64_VG1x4_I64 : Inst<"sv" # n_suffix # "_write_za64[_{d}]_vg1x4", "vm44", "lUl", MergeNone, "aarch64_sme_" # n_suffix # "_write_za_vg1x4", [IsStreaming, IsSharedZA], []>;

      def NAME # _ZA64_VG1X2_I64 : Inst<"sv" # n_suffix # "_za64[_{d}]_vg1x2", "vm2", "lUl", MergeNone, "aarch64_sme_" # n_suffix # "_za64_vg1x2", [IsStreaming, IsSharedZA], []>;
      def NAME # _ZA64_VG1X4_I64 : Inst<"sv" # n_suffix # "_za64[_{d}]_vg1x4", "vm4", "lUl", MergeNone, "aarch64_sme_" # n_suffix # "_za64_vg1x4", [IsStreaming, IsSharedZA], []>;
    }

    let TargetGuard = "sme-f64f64" in {
      def NAME # _ZA64_VG1X2_F64 : Inst<"sv" # n_suffix # "_za64[_{d}]_vg1x2", "vm2", "d", MergeNone, "aarch64_sme_" # n_suffix # "_za64_vg1x2", [IsStreaming, IsSharedZA], []>;
      def NAME # _ZA64_VG1X4_F64 : Inst<"sv" # n_suffix # "_za64[_{d}]_vg1x4", "vm4", "d", MergeNone, "aarch64_sme_" # n_suffix # "_za64_vg1x4", [IsStreaming, IsSharedZA], []>;
    }
  }
}

defm SVADD : ZAAddSub<"add">;
defm SVSUB : ZAAddSub<"sub">;

// SME2 - MOVA

//
// Single, 2 and 4 vector-group read/write intrinsics.
//

multiclass ZAWrite_VG<string n, string t, string i, list<ImmCheck> checks> {
  def NAME # _VG2_H : Inst<"svwrite_hor_" # n # "[_{d}]_vg2",   "vim2", t, MergeNone, i # "_hor_vg2", [IsSharedZA, IsStreaming], checks>;
  def NAME # _VG2_V : Inst<"svwrite_ver_" # n # "[_{d}]_vg2",   "vim2", t, MergeNone, i # "_ver_vg2", [IsSharedZA, IsStreaming], checks>;
  def NAME # _VG4_H : Inst<"svwrite_hor_" # n # "[_{d}]_vg4",   "vim4", t, MergeNone, i # "_hor_vg4", [IsSharedZA, IsStreaming], checks>;
  def NAME # _VG4_V : Inst<"svwrite_ver_" # n # "[_{d}]_vg4",   "vim4", t, MergeNone, i # "_ver_vg4", [IsSharedZA, IsStreaming], checks>;
  def NAME # _VG1x2 : Inst<"svwrite_"     # n # "[_{d}]_vg1x2", "vm2",  t, MergeNone, i # "_vg1x2",   [IsSharedZA, IsStreaming], []>;
  def NAME # _VG1x4 : Inst<"svwrite_"     # n # "[_{d}]_vg1x4", "vm4",  t, MergeNone, i # "_vg1x4",   [IsSharedZA, IsStreaming], []>;
}

let TargetGuard = "sme2" in {
  defm SVWRITE_ZA8  : ZAWrite_VG<"za8",  "cUc",   "aarch64_sme_write", [ImmCheck<0, ImmCheck0_0>]>;
  defm SVWRITE_ZA16 : ZAWrite_VG<"za16", "sUshb", "aarch64_sme_write", [ImmCheck<0, ImmCheck0_1>]>;
  defm SVWRITE_ZA32 : ZAWrite_VG<"za32", "iUif",  "aarch64_sme_write", [ImmCheck<0, ImmCheck0_3>]>;
  defm SVWRITE_ZA64 : ZAWrite_VG<"za64", "lUld",  "aarch64_sme_write", [ImmCheck<0, ImmCheck0_7>]>;
}

multiclass ZARead_VG<string n, string t, string i, list<ImmCheck> checks> {
  def NAME # _VG2_H : Inst<"svread_hor_" # n # "_{d}_vg2",   "2im", t, MergeNone, i # "_hor_vg2", [IsSharedZA, IsPreservesZA, IsStreaming], checks>;
  def NAME # _VG2_V : Inst<"svread_ver_" # n # "_{d}_vg2",   "2im", t, MergeNone, i # "_ver_vg2", [IsSharedZA, IsPreservesZA, IsStreaming], checks>;
  def NAME # _VG4_H : Inst<"svread_hor_" # n # "_{d}_vg4",   "4im", t, MergeNone, i # "_hor_vg4", [IsSharedZA, IsPreservesZA, IsStreaming], checks>;
  def NAME # _VG4_V : Inst<"svread_ver_" # n # "_{d}_vg4",   "4im", t, MergeNone, i # "_ver_vg4", [IsSharedZA, IsPreservesZA, IsStreaming], checks>;
  def NAME # _VG1x2 : Inst<"svread_"     # n # "_{d}_vg1x2", "2m",  t, MergeNone, i # "_vg1x2",   [IsSharedZA, IsPreservesZA, IsStreaming], []>;
  def NAME # _VG1x4 : Inst<"svread_"     # n # "_{d}_vg1x4", "4m",  t, MergeNone, i # "_vg1x4",   [IsSharedZA, IsPreservesZA, IsStreaming], []>;
}

let TargetGuard = "sme2" in {
  defm SVREAD_ZA8  : ZARead_VG<"za8",  "cUc",   "aarch64_sme_read", [ImmCheck<0, ImmCheck0_0>]>;
  defm SVREAD_ZA16 : ZARead_VG<"za16", "sUshb", "aarch64_sme_read", [ImmCheck<0, ImmCheck0_1>]>;
  defm SVREAD_ZA32 : ZARead_VG<"za32", "iUif",  "aarch64_sme_read", [ImmCheck<0, ImmCheck0_3>]>;
  defm SVREAD_ZA64 : ZARead_VG<"za64", "lUld",  "aarch64_sme_read", [ImmCheck<0, ImmCheck0_7>]>;
}

//
// Outer product and accumulate/subtract
//

let TargetGuard = "sme2" in {
  def SVSMOPA  : Inst<"svmopa_za32[_{d}]_m", "viPPdd", "s", MergeNone, "aarch64_sme_smopa_za32", [IsSharedZA, IsStreaming], [ImmCheck<0, ImmCheck0_3>]>;
  def SVUSMOPA : Inst<"svmopa_za32[_{d}]_m", "viPPdd", "Us", MergeNone, "aarch64_sme_umopa_za32", [IsSharedZA, IsStreaming], [ImmCheck<0, ImmCheck0_3>]>;

  def SVSMOPS  : Inst<"svmops_za32[_{d}]_m", "viPPdd", "s", MergeNone, "aarch64_sme_smops_za32", [IsSharedZA, IsStreaming], [ImmCheck<0, ImmCheck0_3>]>;
  def SVUSMOPS : Inst<"svmops_za32[_{d}]_m", "viPPdd", "Us", MergeNone, "aarch64_sme_umops_za32", [IsSharedZA, IsStreaming], [ImmCheck<0, ImmCheck0_3>]>;

  def SVBMOPA : Inst<"svbmopa_za32[_{d}]_m", "viPPdd", "iUi", MergeNone, "aarch64_sme_bmopa_za32", [IsSharedZA, IsStreaming], [ImmCheck<0, ImmCheck0_3>]>;

  def SVBMOPS : Inst<"svbmops_za32[_{d}]_m", "viPPdd", "iUi", MergeNone, "aarch64_sme_bmops_za32", [IsSharedZA, IsStreaming], [ImmCheck<0, ImmCheck0_3>]>;
}

//
// Spill and fill of ZT0
//
let TargetGuard = "sme2" in {
  def SVLDR_ZT : Inst<"svldr_zt", "viQ", "", MergeNone, "aarch64_sme_ldr_zt", [IsOverloadNone, IsStreamingCompatible, IsSharedZA], [ImmCheck<0, ImmCheck0_0>]>;
  def SVSTR_ZT : Inst<"svstr_zt", "vi%", "", MergeNone, "aarch64_sme_str_zt", [IsOverloadNone, IsStreamingCompatible, IsSharedZA, IsPreservesZA], [ImmCheck<0, ImmCheck0_0>]>;
}

//
// Zero ZT0
//
let TargetGuard = "sme2" in {
  def SVZERO_ZT : Inst<"svzero_zt", "vi", "", MergeNone, "aarch64_sme_zero_zt", [IsOverloadNone, IsStreamingCompatible, IsSharedZA], [ImmCheck<0, ImmCheck0_0>]>;
}

//
// lookup table expand four contiguous registers
//
let TargetGuard = "sme2" in {
  def SVLUTI2_LANE_ZT_X4 : Inst<"svluti2_lane_zt_{d}_x4", "4.di[i", "cUcsUsiUibhf", MergeNone, "aarch64_sme_luti2_lane_zt_x4", [IsStreaming, IsSharedZA, IsPreservesZA], [ImmCheck<0, ImmCheck0_0>, ImmCheck<2, ImmCheck0_3>]>;
  def SVLUTI4_LANE_ZT_X4 : Inst<"svluti4_lane_zt_{d}_x4", "4.di[i", "sUsiUibhf", MergeNone, "aarch64_sme_luti4_lane_zt_x4", [IsStreaming, IsSharedZA, IsPreservesZA], [ImmCheck<0, ImmCheck0_0>, ImmCheck<2, ImmCheck0_1>]>;
}

//
// lookup table expand one register
//
let TargetGuard = "sme2" in {
  def SVLUTI2_LANE_ZT : Inst<"svluti2_lane_zt_{d}", "di[i", "cUcsUsiUibhf", MergeNone, "aarch64_sme_luti2_lane_zt", [IsStreaming, IsSharedZA, IsPreservesZA], [ImmCheck<0, ImmCheck0_0>, ImmCheck<2, ImmCheck0_15>]>;
  def SVLUTI4_LANE_ZT : Inst<"svluti4_lane_zt_{d}", "di[i", "cUcsUsiUibhf", MergeNone, "aarch64_sme_luti4_lane_zt", [IsStreaming, IsSharedZA, IsPreservesZA], [ImmCheck<0, ImmCheck0_0>, ImmCheck<2, ImmCheck0_7>]>;
}

//
// lookup table expand two contiguous registers
//
let TargetGuard = "sme2" in {
  def SVLUTI2_LANE_ZT_X2 : Inst<"svluti2_lane_zt_{d}_x2", "2.di[i", "cUcsUsiUibhf", MergeNone, "aarch64_sme_luti2_lane_zt_x2", [IsStreaming, IsSharedZA, IsPreservesZA], [ImmCheck<0, ImmCheck0_0>, ImmCheck<2, ImmCheck0_7>]>;
  def SVLUTI4_LANE_ZT_X2 : Inst<"svluti4_lane_zt_{d}_x2", "2.di[i", "cUcsUsiUibhf", MergeNone, "aarch64_sme_luti4_lane_zt_x2", [IsStreaming, IsSharedZA, IsPreservesZA], [ImmCheck<0, ImmCheck0_0>, ImmCheck<2, ImmCheck0_3>]>;
}
