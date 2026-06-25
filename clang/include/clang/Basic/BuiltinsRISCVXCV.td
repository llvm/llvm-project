//==- BuiltinsRISCVXCV.td - RISC-V CORE-V Builtin database    ----*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the CORE-V-specific builtin function database.  Users of
// this file must define the BUILTIN macro to make use of this information.
//
//===----------------------------------------------------------------------===//

class RISCVXCVBuiltin<string prototype, string features = ""> : TargetBuiltin {
  let Spellings = ["__builtin_riscv_cv_" # NAME];
  let Prototype = prototype;
  let Features = features;
}

let Attributes = [NoThrow, Const] in {
//===----------------------------------------------------------------------===//
// XCValu extension.
//===----------------------------------------------------------------------===//
def alu_sle   : RISCVXCVBuiltin<"int(int, int)", "xcvalu">;
def alu_sleu  : RISCVXCVBuiltin<"int(unsigned int, unsigned int)", "xcvalu">;
def alu_exths : RISCVXCVBuiltin<"int(int)", "xcvalu">;
def alu_exthz : RISCVXCVBuiltin<"unsigned int(unsigned int)", "xcvalu">;
def alu_extbs : RISCVXCVBuiltin<"int(int)", "xcvalu">;
def alu_extbz : RISCVXCVBuiltin<"unsigned int(unsigned int)", "xcvalu">;

def alu_clip   : RISCVXCVBuiltin<"int(int, int)", "xcvalu">;
def alu_clipu  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvalu">;
def alu_addN   : RISCVXCVBuiltin<"int(int, int, unsigned int)", "xcvalu">;
def alu_adduN  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", "xcvalu">;
def alu_addRN  : RISCVXCVBuiltin<"int(int, int, unsigned int)", "xcvalu">;
def alu_adduRN : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", "xcvalu">;
def alu_subN   : RISCVXCVBuiltin<"int(int, int, unsigned int)", "xcvalu">;
def alu_subuN  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", "xcvalu">;
def alu_subRN  : RISCVXCVBuiltin<"int(int, int, unsigned int)", "xcvalu">;
def alu_subuRN : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", "xcvalu">;
} // Attributes = [NoThrow, Const]

//===----------------------------------------------------------------------===//
// XCVbitmanip extension.
//===----------------------------------------------------------------------===//
let Attributes = [NoThrow, Const] in {
def bitmanip_extract  : RISCVXCVBuiltin<"int(unsigned int, unsigned int)", "xcvbitmanip">;
def bitmanip_extractu : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvbitmanip">;
def bitmanip_bclr     : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvbitmanip">;
def bitmanip_bset     : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvbitmanip">;
def bitmanip_insert   : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", "xcvbitmanip">;
def bitmanip_clb      : RISCVXCVBuiltin<"unsigned int(unsigned int)", "xcvbitmanip">;
def bitmanip_bitrev   : RISCVXCVBuiltin<"unsigned int(unsigned int, _Constant unsigned int, _Constant unsigned int)", "xcvbitmanip">;
} // Attributes = [NoThrow, Const]

//===----------------------------------------------------------------------===//
// XCVelw extension.
//===----------------------------------------------------------------------===//
let Attributes = [NoThrow] in {
def elw_elw : RISCVXCVBuiltin<"int(int *)", "xcvelw">;
} // Attributes = [NoThrow]

//===----------------------------------------------------------------------===//
// XCVmac extension.
//===----------------------------------------------------------------------===//
let Attributes = [NoThrow, Const] in {
def mac_mac      : RISCVXCVBuiltin<"int(int, int, int)", "xcvmac">;
def mac_msu      : RISCVXCVBuiltin<"int(int, int, int)", "xcvmac">;
def mac_muluN    : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, _Constant unsigned int)", "xcvmac">;
def mac_mulhhuN  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, _Constant unsigned int)", "xcvmac">;
def mac_mulsN    : RISCVXCVBuiltin<"int(unsigned int, unsigned int, _Constant unsigned int)", "xcvmac">;
def mac_mulhhsN  : RISCVXCVBuiltin<"int(unsigned int, unsigned int, _Constant unsigned int)", "xcvmac">;
def mac_muluRN   : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, _Constant unsigned int)", "xcvmac">;
def mac_mulhhuRN : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, _Constant unsigned int)", "xcvmac">;
def mac_mulsRN   : RISCVXCVBuiltin<"int(unsigned int, unsigned int, _Constant unsigned int)", "xcvmac">;
def mac_mulhhsRN : RISCVXCVBuiltin<"int(unsigned int, unsigned int, _Constant unsigned int)", "xcvmac">;
def mac_macuN    : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int, _Constant unsigned int)", "xcvmac">;
def mac_machhuN  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int, _Constant unsigned int)", "xcvmac">;
def mac_macsN    : RISCVXCVBuiltin<"int(unsigned int, unsigned int, unsigned int, _Constant unsigned int)", "xcvmac">;
def mac_machhsN  : RISCVXCVBuiltin<"int(unsigned int, unsigned int, unsigned int, _Constant unsigned int)", "xcvmac">;
def mac_macuRN   : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int, _Constant unsigned int)", "xcvmac">;
def mac_machhuRN : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int, _Constant unsigned int)", "xcvmac">;
def mac_macsRN   : RISCVXCVBuiltin<"int(unsigned int, unsigned int, unsigned int, _Constant unsigned int)", "xcvmac">;
def mac_machhsRN : RISCVXCVBuiltin<"int(unsigned int, unsigned int, unsigned int, _Constant unsigned int)", "xcvmac">;
} // Attributes = [NoThrow, Const]

//===----------------------------------------------------------------------===//
// XCVsimd extension.
//===----------------------------------------------------------------------===//
let Attributes = [NoThrow, Const] in {

// Element-wise ops are expressed as native packed-vector IR (2 x i16 /
// 4 x i8) so they select the cv.*.h / cv.*.b instructions directly, with no
// dedicated intrinsic. The packed value is carried in a 32-bit GPR.
// (The scalar-broadcast `_sc`/`_sci` forms, the scaled add/sub `div` forms,
// shifts, avg and the comparisons are added later, with their codegen.)

// ADD / SUB
def simd_add_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_add_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_sub_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_sub_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;

// MIN / MINU / MAX / MAXU
def simd_min_h  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_min_b  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_minu_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_minu_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_max_h  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_max_b  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_maxu_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_maxu_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;

// AND / OR / XOR
def simd_and_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_and_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_or_h  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_or_b  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_xor_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_xor_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;

// ABS
def simd_abs_h : RISCVXCVBuiltin<"unsigned int(unsigned int)", "xcvsimd">;
def simd_abs_b : RISCVXCVBuiltin<"unsigned int(unsigned int)", "xcvsimd">;

// DOT PRODUCTS
def simd_dotup_h  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_dotup_b  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_dotup_sc_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_dotup_sc_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_dotusp_h : RISCVXCVBuiltin<"int(unsigned int, unsigned int)", "xcvsimd">;
def simd_dotusp_b : RISCVXCVBuiltin<"int(unsigned int, unsigned int)", "xcvsimd">;
def simd_dotusp_sc_h : RISCVXCVBuiltin<"int(unsigned int, unsigned int)", "xcvsimd">;
def simd_dotusp_sc_b : RISCVXCVBuiltin<"int(unsigned int, unsigned int)", "xcvsimd">;
def simd_dotsp_h : RISCVXCVBuiltin<"int(unsigned int, unsigned int)", "xcvsimd">;
def simd_dotsp_b : RISCVXCVBuiltin<"int(unsigned int, unsigned int)", "xcvsimd">;
def simd_dotsp_sc_h : RISCVXCVBuiltin<"int(unsigned int, unsigned int)", "xcvsimd">;
def simd_dotsp_sc_b : RISCVXCVBuiltin<"int(unsigned int, unsigned int)", "xcvsimd">;

// SDOT (accumulating dot products — rd is both input and output)
def simd_sdotup_h  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", "xcvsimd">;
def simd_sdotup_b  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", "xcvsimd">;
def simd_sdotup_sc_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", "xcvsimd">;
def simd_sdotup_sc_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", "xcvsimd">;
def simd_sdotusp_h : RISCVXCVBuiltin<"int(unsigned int, unsigned int, int)", "xcvsimd">;
def simd_sdotusp_b : RISCVXCVBuiltin<"int(unsigned int, unsigned int, int)", "xcvsimd">;
def simd_sdotusp_sc_h : RISCVXCVBuiltin<"int(unsigned int, unsigned int, int)", "xcvsimd">;
def simd_sdotusp_sc_b : RISCVXCVBuiltin<"int(unsigned int, unsigned int, int)", "xcvsimd">;
def simd_sdotsp_h : RISCVXCVBuiltin<"int(unsigned int, unsigned int, int)", "xcvsimd">;
def simd_sdotsp_b : RISCVXCVBuiltin<"int(unsigned int, unsigned int, int)", "xcvsimd">;
def simd_sdotsp_sc_h : RISCVXCVBuiltin<"int(unsigned int, unsigned int, int)", "xcvsimd">;
def simd_sdotsp_sc_b : RISCVXCVBuiltin<"int(unsigned int, unsigned int, int)", "xcvsimd">;

// EXTRACT / INSERT
def simd_extract_h  : RISCVXCVBuiltin<"int(unsigned int, _Constant unsigned int)", "xcvsimd">;
def simd_extract_b  : RISCVXCVBuiltin<"int(unsigned int, _Constant unsigned int)", "xcvsimd">;
def simd_extractu_h : RISCVXCVBuiltin<"unsigned int(unsigned int, _Constant unsigned int)", "xcvsimd">;
def simd_extractu_b : RISCVXCVBuiltin<"unsigned int(unsigned int, _Constant unsigned int)", "xcvsimd">;
def simd_insert_h   : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, _Constant unsigned int)", "xcvsimd">;
def simd_insert_b   : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, _Constant unsigned int)", "xcvsimd">;

// SHUFFLE / SHUFFLE2
def simd_shuffle_h    : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_shuffle_b    : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_shuffle_sci_h : RISCVXCVBuiltin<"unsigned int(unsigned int, _Constant unsigned int)", "xcvsimd">;
def simd_shuffle_sci_b : RISCVXCVBuiltin<"unsigned int(unsigned int, _Constant unsigned int)", "xcvsimd">;
def simd_shuffle2_h   : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", "xcvsimd">;
def simd_shuffle2_b   : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", "xcvsimd">;

// PACK
def simd_packhi_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_packlo_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_packhi_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", "xcvsimd">;
def simd_packlo_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", "xcvsimd">;

// COMPLEX NUMBER OPS
def simd_cplxmul_r : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int, _Constant unsigned int)", "xcvsimd">;
def simd_cplxmul_i : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int, _Constant unsigned int)", "xcvsimd">;
def simd_cplxconj   : RISCVXCVBuiltin<"unsigned int(unsigned int)", "xcvsimd">;
def simd_subrotmj   : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, _Constant unsigned int)", "xcvsimd">;

} // Attributes = [NoThrow, Const]
