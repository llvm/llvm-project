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
    def NAME # _H : MInst<"svld1_hor_" # n_suffix, "vimiPQ", t,
                          [IsLoad, IsOverloadNone, IsStreaming, IsSharedZA],
                          MemEltTyDefault, i_prefix # "_horiz", ch>;

    def NAME # _H_VNUM : MInst<"svld1_hor_vnum_" # n_suffix, "vimiPQl", t,
                               [IsLoad, IsOverloadNone, IsStreaming, IsSharedZA],
                               MemEltTyDefault, i_prefix # "_horiz", ch>;

    def NAME # _V : MInst<"svld1_ver_" # n_suffix, "vimiPQ", t,
                          [IsLoad, IsOverloadNone, IsStreaming, IsSharedZA],
                          MemEltTyDefault, i_prefix # "_vert", ch>;

    def NAME # _V_VNUM : MInst<"svld1_ver_vnum_" # n_suffix, "vimiPQl", t,
                               [IsLoad, IsOverloadNone, IsStreaming, IsSharedZA],
                               MemEltTyDefault, i_prefix # "_vert", ch>;
  }
}

defm SVLD1_ZA8 : ZALoad<"za8", "c", "aarch64_sme_ld1b", [ImmCheck<0, ImmCheck0_0>, ImmCheck<2, ImmCheck0_15>]>;
defm SVLD1_ZA16 : ZALoad<"za16", "s", "aarch64_sme_ld1h", [ImmCheck<0, ImmCheck0_1>, ImmCheck<2, ImmCheck0_7>]>;
defm SVLD1_ZA32 : ZALoad<"za32", "i", "aarch64_sme_ld1w", [ImmCheck<0, ImmCheck0_3>, ImmCheck<2, ImmCheck0_3>]>;
defm SVLD1_ZA64 : ZALoad<"za64", "l", "aarch64_sme_ld1d", [ImmCheck<0, ImmCheck0_7>, ImmCheck<2, ImmCheck0_1>]>;
defm SVLD1_ZA128 : ZALoad<"za128", "q", "aarch64_sme_ld1q", [ImmCheck<0, ImmCheck0_15>, ImmCheck<2, ImmCheck0_0>]>;

////////////////////////////////////////////////////////////////////////////////
// Stores

multiclass ZAStore<string n_suffix, string t, string i_prefix, list<ImmCheck> ch> {
  let TargetGuard = "sme" in {
    def NAME # _H : MInst<"svst1_hor_" # n_suffix, "vimiP%", t,
                          [IsStore, IsOverloadNone, IsStreaming, IsSharedZA, IsPreservesZA],
                          MemEltTyDefault, i_prefix # "_horiz", ch>;

    def NAME # _H_VNUM : MInst<"svst1_hor_vnum_" # n_suffix, "vimiP%l", t,
                               [IsStore, IsOverloadNone, IsStreaming, IsSharedZA, IsPreservesZA],
                               MemEltTyDefault, i_prefix # "_horiz", ch>;

    def NAME # _V : MInst<"svst1_ver_" # n_suffix, "vimiP%", t,
                          [IsStore, IsOverloadNone, IsStreaming, IsSharedZA, IsPreservesZA],
                          MemEltTyDefault, i_prefix # "_vert", ch>;

    def NAME # _V_VNUM : MInst<"svst1_ver_vnum_" # n_suffix, "vimiP%l", t,
                               [IsStore, IsOverloadNone, IsStreaming, IsSharedZA, IsPreservesZA],
                               MemEltTyDefault, i_prefix # "_vert", ch>;
  }
}

defm SVST1_ZA8 : ZAStore<"za8", "c", "aarch64_sme_st1b", [ImmCheck<0, ImmCheck0_0>, ImmCheck<2, ImmCheck0_15>]>;
defm SVST1_ZA16 : ZAStore<"za16", "s", "aarch64_sme_st1h", [ImmCheck<0, ImmCheck0_1>, ImmCheck<2, ImmCheck0_7>]>;
defm SVST1_ZA32 : ZAStore<"za32", "i", "aarch64_sme_st1w", [ImmCheck<0, ImmCheck0_3>, ImmCheck<2, ImmCheck0_3>]>;
defm SVST1_ZA64 : ZAStore<"za64", "l", "aarch64_sme_st1d", [ImmCheck<0, ImmCheck0_7>, ImmCheck<2, ImmCheck0_1>]>;
defm SVST1_ZA128 : ZAStore<"za128", "q", "aarch64_sme_st1q", [ImmCheck<0, ImmCheck0_15>, ImmCheck<2, ImmCheck0_0>]>;

////////////////////////////////////////////////////////////////////////////////
// Read horizontal/vertical ZA slices

multiclass ZARead<string n_suffix, string t, string i_prefix, list<ImmCheck> ch> {
  let TargetGuard = "sme" in {
    def NAME # _H : SInst<"svread_hor_" # n_suffix # "[_{d}]", "ddPimi", t,
                          MergeOp1, i_prefix # "_horiz",
                          [IsReadZA, IsStreaming, IsSharedZA, IsPreservesZA], ch>;

    def NAME # _V : SInst<"svread_ver_" # n_suffix # "[_{d}]", "ddPimi", t,
                          MergeOp1, i_prefix # "_vert",
                          [IsReadZA, IsStreaming, IsSharedZA, IsPreservesZA], ch>;
  }
}

defm SVREAD_ZA8 : ZARead<"za8", "cUc", "aarch64_sme_read", [ImmCheck<2, ImmCheck0_0>, ImmCheck<4, ImmCheck0_15>]>;
defm SVREAD_ZA16 : ZARead<"za16", "sUshb", "aarch64_sme_read", [ImmCheck<2, ImmCheck0_1>, ImmCheck<4, ImmCheck0_7>]>;
defm SVREAD_ZA32 : ZARead<"za32", "iUif", "aarch64_sme_read", [ImmCheck<2, ImmCheck0_3>, ImmCheck<4, ImmCheck0_3>]>;
defm SVREAD_ZA64 : ZARead<"za64", "lUld", "aarch64_sme_read", [ImmCheck<2, ImmCheck0_7>, ImmCheck<4, ImmCheck0_1>]>;
defm SVREAD_ZA128 : ZARead<"za128", "csilUcUsUiUlhbfd", "aarch64_sme_readq", [ImmCheck<2, ImmCheck0_15>, ImmCheck<4, ImmCheck0_0>]>;

////////////////////////////////////////////////////////////////////////////////
// Write horizontal/vertical ZA slices

multiclass ZAWrite<string n_suffix, string t, string i_prefix, list<ImmCheck> ch> {
  let TargetGuard = "sme" in {
    def NAME # _H : SInst<"svwrite_hor_" # n_suffix # "[_{d}]", "vimiPd", t,
                          MergeOp1, i_prefix # "_horiz",
                          [IsWriteZA, IsStreaming, IsSharedZA], ch>;

    def NAME # _V : SInst<"svwrite_ver_" # n_suffix # "[_{d}]", "vimiPd", t,
                          MergeOp1, i_prefix # "_vert",
                          [IsWriteZA, IsStreaming, IsSharedZA], ch>;
  }
}

defm SVWRITE_ZA8 : ZAWrite<"za8", "cUc", "aarch64_sme_write", [ImmCheck<0, ImmCheck0_0>, ImmCheck<2, ImmCheck0_15>]>;
defm SVWRITE_ZA16 : ZAWrite<"za16", "sUshb", "aarch64_sme_write", [ImmCheck<0, ImmCheck0_1>, ImmCheck<2, ImmCheck0_7>]>;
defm SVWRITE_ZA32 : ZAWrite<"za32", "iUif", "aarch64_sme_write", [ImmCheck<0, ImmCheck0_3>, ImmCheck<2, ImmCheck0_3>]>;
defm SVWRITE_ZA64 : ZAWrite<"za64", "lUld", "aarch64_sme_write", [ImmCheck<0, ImmCheck0_7>, ImmCheck<2, ImmCheck0_1>]>;
defm SVWRITE_ZA128 : ZAWrite<"za128", "csilUcUsUiUlhbfd", "aarch64_sme_writeq", [ImmCheck<0, ImmCheck0_15>, ImmCheck<2, ImmCheck0_0>]>;

////////////////////////////////////////////////////////////////////////////////
// SME - Zero

let TargetGuard = "sme" in {
  def SVZERO_MASK_ZA : SInst<"svzero_mask_za", "vi", "", MergeNone, "aarch64_sme_zero",
                             [IsOverloadNone, IsStreamingCompatible, IsSharedZA],
                             [ImmCheck<0, ImmCheck0_255>]>;
  def SVZERO_ZA      : SInst<"svzero_za", "v", "", MergeNone, "aarch64_sme_zero",
                             [IsOverloadNone, IsStreamingCompatible, IsSharedZA]>;
}
