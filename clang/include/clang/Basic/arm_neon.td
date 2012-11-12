//===--- arm_neon.td - ARM NEON compiler interface ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the TableGen definitions from which the ARM NEON header
//  file will be generated.  See ARM document DUI0348B.
//
//===----------------------------------------------------------------------===//

class Op;

def OP_NONE  : Op;
def OP_UNAVAILABLE : Op;
def OP_ADD   : Op;
def OP_ADDL  : Op;
def OP_ADDW  : Op;
def OP_SUB   : Op;
def OP_SUBL  : Op;
def OP_SUBW  : Op;
def OP_MUL   : Op;
def OP_MLA   : Op;
def OP_MLAL  : Op;
def OP_MLS   : Op;
def OP_MLSL  : Op;
def OP_MUL_N : Op;
def OP_MLA_N : Op;
def OP_MLS_N : Op;
def OP_MLAL_N : Op;
def OP_MLSL_N : Op;
def OP_MUL_LN: Op;
def OP_MULL_LN : Op;
def OP_MLA_LN: Op;
def OP_MLS_LN: Op;
def OP_MLAL_LN : Op;
def OP_MLSL_LN : Op;
def OP_QDMULL_LN : Op;
def OP_QDMLAL_LN : Op;
def OP_QDMLSL_LN : Op;
def OP_QDMULH_LN : Op;
def OP_QRDMULH_LN : Op;
def OP_EQ    : Op;
def OP_GE    : Op;
def OP_LE    : Op;
def OP_GT    : Op;
def OP_LT    : Op;
def OP_NEG   : Op;
def OP_NOT   : Op;
def OP_AND   : Op;
def OP_OR    : Op;
def OP_XOR   : Op;
def OP_ANDN  : Op;
def OP_ORN   : Op;
def OP_CAST  : Op;
def OP_HI    : Op;
def OP_LO    : Op;
def OP_CONC  : Op;
def OP_DUP   : Op;
def OP_DUP_LN: Op;
def OP_SEL   : Op;
def OP_REV64 : Op;
def OP_REV32 : Op;
def OP_REV16 : Op;
def OP_REINT : Op;
def OP_ABDL  : Op;
def OP_ABA   : Op;
def OP_ABAL  : Op;

class Inst <string n, string p, string t, Op o> {
  string Name = n;
  string Prototype = p;
  string Types = t;
  Op Operand = o;
  bit isShift = 0;
  bit isVCVT_N = 0;
}

// Used to generate Builtins.def:
// SInst: Instruction with signed/unsigned suffix (e.g., "s8", "u8", "p8")
// IInst: Instruction with generic integer suffix (e.g., "i8")
// WInst: Instruction with only bit size suffix (e.g., "8")
class SInst<string n, string p, string t> : Inst<n, p, t, OP_NONE> {}
class IInst<string n, string p, string t> : Inst<n, p, t, OP_NONE> {}
class WInst<string n, string p, string t> : Inst<n, p, t, OP_NONE> {}

// prototype: return (arg, arg, ...)
// v: void
// t: best-fit integer (int/poly args)
// x: signed integer   (int/float args)
// u: unsigned integer (int/float args)
// f: float (int args)
// d: default
// g: default, ignore 'Q' size modifier.
// w: double width elements, same num elts
// n: double width elements, half num elts
// h: half width elements, double num elts
// e: half width elements, double num elts, unsigned
// i: constant int
// l: constant uint64
// s: scalar of element type
// a: scalar of element type (splat to vector type)
// k: default elt width, double num elts
// #: array of default vectors
// p: pointer type
// c: const pointer type

// sizes:
// c: char
// s: short
// i: int
// l: long
// f: float
// h: half-float

// size modifiers:
// U: unsigned
// Q: 128b
// P: polynomial

////////////////////////////////////////////////////////////////////////////////
// E.3.1 Addition
def VADD : Inst<"vadd", "ddd", "csilfUcUsUiUlQcQsQiQlQfQUcQUsQUiQUl", OP_ADD>;
def VADDL   : Inst<"vaddl", "wdd", "csiUcUsUi", OP_ADDL>;
def VADDW   : Inst<"vaddw", "wwd", "csiUcUsUi", OP_ADDW>;
def VHADD   : SInst<"vhadd", "ddd", "csiUcUsUiQcQsQiQUcQUsQUi">;
def VRHADD  : SInst<"vrhadd", "ddd", "csiUcUsUiQcQsQiQUcQUsQUi">;
def VQADD   : SInst<"vqadd", "ddd", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VADDHN  : IInst<"vaddhn", "hkk", "silUsUiUl">;
def VRADDHN : IInst<"vraddhn", "hkk", "silUsUiUl">;

////////////////////////////////////////////////////////////////////////////////
// E.3.2 Multiplication
def VMUL     : Inst<"vmul", "ddd", "csifUcUsUiQcQsQiQfQUcQUsQUi", OP_MUL>;
def VMULP    : SInst<"vmul", "ddd", "PcQPc">;
def VMLA     : Inst<"vmla", "dddd", "csifUcUsUiQcQsQiQfQUcQUsQUi", OP_MLA>;
def VMLAL    : Inst<"vmlal", "wwdd", "csiUcUsUi", OP_MLAL>;
def VMLS     : Inst<"vmls", "dddd", "csifUcUsUiQcQsQiQfQUcQUsQUi", OP_MLS>;
def VMLSL    : Inst<"vmlsl", "wwdd", "csiUcUsUi", OP_MLSL>;
def VQDMULH  : SInst<"vqdmulh", "ddd", "siQsQi">;
def VQRDMULH : SInst<"vqrdmulh", "ddd", "siQsQi">;
def VQDMLAL  : SInst<"vqdmlal", "wwdd", "si">;
def VQDMLSL  : SInst<"vqdmlsl", "wwdd", "si">;
def VMULL    : SInst<"vmull", "wdd", "csiUcUsUiPc">;
def VQDMULL  : SInst<"vqdmull", "wdd", "si">;

////////////////////////////////////////////////////////////////////////////////
// E.3.3 Subtraction
def VSUB : Inst<"vsub", "ddd", "csilfUcUsUiUlQcQsQiQlQfQUcQUsQUiQUl", OP_SUB>;
def VSUBL   : Inst<"vsubl", "wdd", "csiUcUsUi", OP_SUBL>;
def VSUBW   : Inst<"vsubw", "wwd", "csiUcUsUi", OP_SUBW>;
def VQSUB   : SInst<"vqsub", "ddd", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VHSUB   : SInst<"vhsub", "ddd", "csiUcUsUiQcQsQiQUcQUsQUi">;
def VSUBHN  : IInst<"vsubhn", "hkk", "silUsUiUl">;
def VRSUBHN : IInst<"vrsubhn", "hkk", "silUsUiUl">;

////////////////////////////////////////////////////////////////////////////////
// E.3.4 Comparison
def VCEQ  : Inst<"vceq", "udd", "csifUcUsUiPcQcQsQiQfQUcQUsQUiQPc", OP_EQ>;
def VCGE  : Inst<"vcge", "udd", "csifUcUsUiQcQsQiQfQUcQUsQUi", OP_GE>;
def VCLE  : Inst<"vcle", "udd", "csifUcUsUiQcQsQiQfQUcQUsQUi", OP_LE>;
def VCGT  : Inst<"vcgt", "udd", "csifUcUsUiQcQsQiQfQUcQUsQUi", OP_GT>;
def VCLT  : Inst<"vclt", "udd", "csifUcUsUiQcQsQiQfQUcQUsQUi", OP_LT>;
def VCAGE : IInst<"vcage", "udd", "fQf">;
def VCALE : IInst<"vcale", "udd", "fQf">;
def VCAGT : IInst<"vcagt", "udd", "fQf">;
def VCALT : IInst<"vcalt", "udd", "fQf">;
def VTST  : WInst<"vtst", "udd", "csiUcUsUiPcQcQsQiQUcQUsQUiQPc">;

////////////////////////////////////////////////////////////////////////////////
// E.3.5 Absolute Difference
def VABD  : SInst<"vabd", "ddd",  "csiUcUsUifQcQsQiQUcQUsQUiQf">;
def VABDL : Inst<"vabdl", "wdd",  "csiUcUsUi", OP_ABDL>;
def VABA  : Inst<"vaba", "dddd", "csiUcUsUiQcQsQiQUcQUsQUi", OP_ABA>;
def VABAL : Inst<"vabal", "wwdd", "csiUcUsUi", OP_ABAL>;

////////////////////////////////////////////////////////////////////////////////
// E.3.6 Max/Min
def VMAX : SInst<"vmax", "ddd", "csiUcUsUifQcQsQiQUcQUsQUiQf">;
def VMIN : SInst<"vmin", "ddd", "csiUcUsUifQcQsQiQUcQUsQUiQf">;

////////////////////////////////////////////////////////////////////////////////
// E.3.7 Pairwise Addition
def VPADD  : IInst<"vpadd", "ddd", "csiUcUsUif">;
def VPADDL : SInst<"vpaddl", "nd",  "csiUcUsUiQcQsQiQUcQUsQUi">;
def VPADAL : SInst<"vpadal", "nnd", "csiUcUsUiQcQsQiQUcQUsQUi">;

////////////////////////////////////////////////////////////////////////////////
// E.3.8-9 Folding Max/Min
def VPMAX : SInst<"vpmax", "ddd", "csiUcUsUif">;
def VPMIN : SInst<"vpmin", "ddd", "csiUcUsUif">;

////////////////////////////////////////////////////////////////////////////////
// E.3.10 Reciprocal/Sqrt
def VRECPS  : IInst<"vrecps", "ddd", "fQf">;
def VRSQRTS : IInst<"vrsqrts", "ddd", "fQf">;

////////////////////////////////////////////////////////////////////////////////
// E.3.11 Shifts by signed variable
def VSHL   : SInst<"vshl", "ddx", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VQSHL  : SInst<"vqshl", "ddx", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VRSHL  : SInst<"vrshl", "ddx", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VQRSHL : SInst<"vqrshl", "ddx", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;

////////////////////////////////////////////////////////////////////////////////
// E.3.12 Shifts by constant
let isShift = 1 in {
def VSHR_N     : SInst<"vshr_n", "ddi", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VSHL_N     : IInst<"vshl_n", "ddi", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VRSHR_N    : SInst<"vrshr_n", "ddi", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VSRA_N     : SInst<"vsra_n", "dddi", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VRSRA_N    : SInst<"vrsra_n", "dddi", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VQSHL_N    : SInst<"vqshl_n", "ddi", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VQSHLU_N   : SInst<"vqshlu_n", "udi", "csilQcQsQiQl">;
def VSHRN_N    : IInst<"vshrn_n", "hki", "silUsUiUl">;
def VQSHRUN_N  : SInst<"vqshrun_n", "eki", "sil">;
def VQRSHRUN_N : SInst<"vqrshrun_n", "eki", "sil">;
def VQSHRN_N   : SInst<"vqshrn_n", "hki", "silUsUiUl">;
def VRSHRN_N   : IInst<"vrshrn_n", "hki", "silUsUiUl">;
def VQRSHRN_N  : SInst<"vqrshrn_n", "hki", "silUsUiUl">;
def VSHLL_N    : SInst<"vshll_n", "wdi", "csiUcUsUi">;

////////////////////////////////////////////////////////////////////////////////
// E.3.13 Shifts with insert
def VSRI_N : WInst<"vsri_n", "dddi",
                   "csilUcUsUiUlPcPsQcQsQiQlQUcQUsQUiQUlQPcQPs">;
def VSLI_N : WInst<"vsli_n", "dddi",
                   "csilUcUsUiUlPcPsQcQsQiQlQUcQUsQUiQUlQPcQPs">;
}

////////////////////////////////////////////////////////////////////////////////
// E.3.14 Loads and stores of a single vector
def VLD1      : WInst<"vld1", "dc",
                      "QUcQUsQUiQUlQcQsQiQlQhQfQPcQPsUcUsUiUlcsilhfPcPs">;
def VLD1_LANE : WInst<"vld1_lane", "dcdi",
                      "QUcQUsQUiQUlQcQsQiQlQhQfQPcQPsUcUsUiUlcsilhfPcPs">;
def VLD1_DUP  : WInst<"vld1_dup", "dc",
                      "QUcQUsQUiQUlQcQsQiQlQhQfQPcQPsUcUsUiUlcsilhfPcPs">;
def VST1      : WInst<"vst1", "vpd",
                      "QUcQUsQUiQUlQcQsQiQlQhQfQPcQPsUcUsUiUlcsilhfPcPs">;
def VST1_LANE : WInst<"vst1_lane", "vpdi",
                      "QUcQUsQUiQUlQcQsQiQlQhQfQPcQPsUcUsUiUlcsilhfPcPs">;

////////////////////////////////////////////////////////////////////////////////
// E.3.15 Loads and stores of an N-element structure
def VLD2 : WInst<"vld2", "2c", "QUcQUsQUiQcQsQiQhQfQPcQPsUcUsUiUlcsilhfPcPs">;
def VLD3 : WInst<"vld3", "3c", "QUcQUsQUiQcQsQiQhQfQPcQPsUcUsUiUlcsilhfPcPs">;
def VLD4 : WInst<"vld4", "4c", "QUcQUsQUiQcQsQiQhQfQPcQPsUcUsUiUlcsilhfPcPs">;
def VLD2_DUP  : WInst<"vld2_dup", "2c", "UcUsUiUlcsilhfPcPs">;
def VLD3_DUP  : WInst<"vld3_dup", "3c", "UcUsUiUlcsilhfPcPs">;
def VLD4_DUP  : WInst<"vld4_dup", "4c", "UcUsUiUlcsilhfPcPs">;
def VLD2_LANE : WInst<"vld2_lane", "2c2i", "QUsQUiQsQiQhQfQPsUcUsUicsihfPcPs">;
def VLD3_LANE : WInst<"vld3_lane", "3c3i", "QUsQUiQsQiQhQfQPsUcUsUicsihfPcPs">;
def VLD4_LANE : WInst<"vld4_lane", "4c4i", "QUsQUiQsQiQhQfQPsUcUsUicsihfPcPs">;
def VST2 : WInst<"vst2", "vp2", "QUcQUsQUiQcQsQiQhQfQPcQPsUcUsUiUlcsilhfPcPs">;
def VST3 : WInst<"vst3", "vp3", "QUcQUsQUiQcQsQiQhQfQPcQPsUcUsUiUlcsilhfPcPs">;
def VST4 : WInst<"vst4", "vp4", "QUcQUsQUiQcQsQiQhQfQPcQPsUcUsUiUlcsilhfPcPs">;
def VST2_LANE : WInst<"vst2_lane", "vp2i", "QUsQUiQsQiQhQfQPsUcUsUicsihfPcPs">;
def VST3_LANE : WInst<"vst3_lane", "vp3i", "QUsQUiQsQiQhQfQPsUcUsUicsihfPcPs">;
def VST4_LANE : WInst<"vst4_lane", "vp4i", "QUsQUiQsQiQhQfQPsUcUsUicsihfPcPs">;

////////////////////////////////////////////////////////////////////////////////
// E.3.16 Extract lanes from a vector
def VGET_LANE : IInst<"vget_lane", "sdi",
                      "UcUsUicsiPcPsfQUcQUsQUiQcQsQiQPcQPsQflUlQlQUl">;

////////////////////////////////////////////////////////////////////////////////
// E.3.17 Set lanes within a vector
def VSET_LANE : IInst<"vset_lane", "dsdi",
                      "UcUsUicsiPcPsfQUcQUsQUiQcQsQiQPcQPsQflUlQlQUl">;

////////////////////////////////////////////////////////////////////////////////
// E.3.18 Initialize a vector from bit pattern
def VCREATE: Inst<"vcreate", "dl", "csihfUcUsUiUlPcPsl", OP_CAST>;

////////////////////////////////////////////////////////////////////////////////
// E.3.19 Set all lanes to same value
def VDUP_N : Inst<"vdup_n", "ds",
                  "UcUsUicsiPcPsfQUcQUsQUiQcQsQiQPcQPsQflUlQlQUl", OP_DUP>;
def VMOV_N : Inst<"vmov_n", "ds",
                  "UcUsUicsiPcPsfQUcQUsQUiQcQsQiQPcQPsQflUlQlQUl", OP_DUP>;
def VDUP_LANE : Inst<"vdup_lane", "dgi",
                     "UcUsUicsiPcPsfQUcQUsQUiQcQsQiQPcQPsQflUlQlQUl",OP_DUP_LN>;

////////////////////////////////////////////////////////////////////////////////
// E.3.20 Combining vectors
def VCOMBINE : Inst<"vcombine", "kdd", "csilhfUcUsUiUlPcPs", OP_CONC>;

////////////////////////////////////////////////////////////////////////////////
// E.3.21 Splitting vectors
def VGET_HIGH : Inst<"vget_high", "dk", "csilhfUcUsUiUlPcPs", OP_HI>;
def VGET_LOW  : Inst<"vget_low", "dk", "csilhfUcUsUiUlPcPs", OP_LO>;

////////////////////////////////////////////////////////////////////////////////
// E.3.22 Converting vectors
def VCVT_S32     : SInst<"vcvt_s32", "xd",  "fQf">;
def VCVT_U32     : SInst<"vcvt_u32", "ud",  "fQf">;
def VCVT_F16     : SInst<"vcvt_f16", "hk",  "f">;
def VCVT_F32     : SInst<"vcvt_f32", "fd",  "iUiQiQUi">;
def VCVT_F32_F16 : SInst<"vcvt_f32_f16", "fd",  "h">;
let isVCVT_N = 1 in {
def VCVT_N_S32   : SInst<"vcvt_n_s32", "xdi", "fQf">;
def VCVT_N_U32   : SInst<"vcvt_n_u32", "udi", "fQf">;
def VCVT_N_F32   : SInst<"vcvt_n_f32", "fdi", "iUiQiQUi">;
}
def VMOVN        : IInst<"vmovn", "hk",  "silUsUiUl">;
def VMOVL        : SInst<"vmovl", "wd",  "csiUcUsUi">;
def VQMOVN       : SInst<"vqmovn", "hk",  "silUsUiUl">;
def VQMOVUN      : SInst<"vqmovun", "ek",  "sil">;

////////////////////////////////////////////////////////////////////////////////
// E.3.23-24 Table lookup, Extended table lookup
def VTBL1 : WInst<"vtbl1", "ddt",  "UccPc">;
def VTBL2 : WInst<"vtbl2", "d2t",  "UccPc">;
def VTBL3 : WInst<"vtbl3", "d3t",  "UccPc">;
def VTBL4 : WInst<"vtbl4", "d4t",  "UccPc">;
def VTBX1 : WInst<"vtbx1", "dddt", "UccPc">;
def VTBX2 : WInst<"vtbx2", "dd2t", "UccPc">;
def VTBX3 : WInst<"vtbx3", "dd3t", "UccPc">;
def VTBX4 : WInst<"vtbx4", "dd4t", "UccPc">;

////////////////////////////////////////////////////////////////////////////////
// E.3.25 Operations with a scalar value
def VMLA_LANE    : Inst<"vmla_lane", "dddgi", "siUsUifQsQiQUsQUiQf", OP_MLA_LN>;
def VMLAL_LANE   : Inst<"vmlal_lane", "wwddi", "siUsUi", OP_MLAL_LN>;
def VQDMLAL_LANE : Inst<"vqdmlal_lane", "wwddi", "si", OP_QDMLAL_LN>; 
def VMLS_LANE    : Inst<"vmls_lane", "dddgi", "siUsUifQsQiQUsQUiQf", OP_MLS_LN>;
def VMLSL_LANE   : Inst<"vmlsl_lane", "wwddi", "siUsUi", OP_MLSL_LN>;
def VQDMLSL_LANE : Inst<"vqdmlsl_lane", "wwddi", "si", OP_QDMLSL_LN>;
def VMUL_N       : Inst<"vmul_n", "dds", "sifUsUiQsQiQfQUsQUi", OP_MUL_N>;
def VMUL_LANE    : Inst<"vmul_lane", "ddgi", "sifUsUiQsQiQfQUsQUi", OP_MUL_LN>;
def VMULL_N      : SInst<"vmull_n", "wda", "siUsUi">;
def VMULL_LANE   : Inst<"vmull_lane", "wddi", "siUsUi", OP_MULL_LN>;
def VQDMULL_N    : SInst<"vqdmull_n", "wda", "si">;
def VQDMULL_LANE : Inst<"vqdmull_lane", "wddi", "si", OP_QDMULL_LN>;
def VQDMULH_N    : SInst<"vqdmulh_n", "dda", "siQsQi">;
def VQDMULH_LANE : Inst<"vqdmulh_lane", "ddgi", "siQsQi", OP_QDMULH_LN>;
def VQRDMULH_N   : SInst<"vqrdmulh_n", "dda", "siQsQi">;
def VQRDMULH_LANE : Inst<"vqrdmulh_lane", "ddgi", "siQsQi", OP_QRDMULH_LN>;
def VMLA_N       : Inst<"vmla_n", "ddda", "siUsUifQsQiQUsQUiQf", OP_MLA_N>;
def VMLAL_N      : Inst<"vmlal_n", "wwda", "siUsUi", OP_MLAL_N>;
def VQDMLAL_N    : SInst<"vqdmlal_n", "wwda", "si">;
def VMLS_N       : Inst<"vmls_n", "ddds", "siUsUifQsQiQUsQUiQf", OP_MLS_N>;
def VMLSL_N      : Inst<"vmlsl_n", "wwda", "siUsUi", OP_MLSL_N>;
def VQDMLSL_N    : SInst<"vqdmlsl_n", "wwda", "si">;

////////////////////////////////////////////////////////////////////////////////
// E.3.26 Vector Extract
def VEXT : WInst<"vext", "dddi",
                 "cUcPcsUsPsiUilUlfQcQUcQPcQsQUsQPsQiQUiQlQUlQf">;

////////////////////////////////////////////////////////////////////////////////
// E.3.27 Reverse vector elements
def VREV64 : Inst<"vrev64", "dd", "csiUcUsUiPcPsfQcQsQiQUcQUsQUiQPcQPsQf",
                  OP_REV64>;
def VREV32 : Inst<"vrev32", "dd", "csUcUsPcPsQcQsQUcQUsQPcQPs", OP_REV32>;
def VREV16 : Inst<"vrev16", "dd", "cUcPcQcQUcQPc", OP_REV16>;

////////////////////////////////////////////////////////////////////////////////
// E.3.28 Other single operand arithmetic
def VABS    : SInst<"vabs", "dd", "csifQcQsQiQf">;
def VQABS   : SInst<"vqabs", "dd", "csiQcQsQi">;
def VNEG    : Inst<"vneg", "dd", "csifQcQsQiQf", OP_NEG>;
def VQNEG   : SInst<"vqneg", "dd", "csiQcQsQi">;
def VCLS    : SInst<"vcls", "dd", "csiQcQsQi">;
def VCLZ    : IInst<"vclz", "dd", "csiUcUsUiQcQsQiQUcQUsQUi">;
def VCNT    : WInst<"vcnt", "dd", "UccPcQUcQcQPc">;
def VRECPE  : SInst<"vrecpe", "dd", "fUiQfQUi">;
def VRSQRTE : SInst<"vrsqrte", "dd", "fUiQfQUi">;

////////////////////////////////////////////////////////////////////////////////
// E.3.29 Logical operations
def VMVN : Inst<"vmvn", "dd", "csiUcUsUiPcQcQsQiQUcQUsQUiQPc", OP_NOT>;
def VAND : Inst<"vand", "ddd", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl", OP_AND>;
def VORR : Inst<"vorr", "ddd", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl", OP_OR>;
def VEOR : Inst<"veor", "ddd", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl", OP_XOR>;
def VBIC : Inst<"vbic", "ddd", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl", OP_ANDN>;
def VORN : Inst<"vorn", "ddd", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl", OP_ORN>;
def VBSL : SInst<"vbsl", "dudd",
                "csilUcUsUiUlfPcPsQcQsQiQlQUcQUsQUiQUlQfQPcQPs">;

////////////////////////////////////////////////////////////////////////////////
// E.3.30 Transposition operations
def VTRN : WInst<"vtrn", "2dd", "csiUcUsUifPcPsQcQsQiQUcQUsQUiQfQPcQPs">;
def VZIP : WInst<"vzip", "2dd", "csiUcUsUifPcPsQcQsQiQUcQUsQUiQfQPcQPs">;
def VUZP : WInst<"vuzp", "2dd", "csiUcUsUifPcPsQcQsQiQUcQUsQUiQfQPcQPs">;

////////////////////////////////////////////////////////////////////////////////
// E.3.31 Vector reinterpret cast operations
def VREINTERPRET
  : Inst<"vreinterpret", "dd",
         "csilUcUsUiUlhfPcPsQcQsQiQlQUcQUsQUiQUlQhQfQPcQPs", OP_REINT>;

////////////////////////////////////////////////////////////////////////////////
// Vector fused multiply-add operations

def VFMA : SInst<"vfma", "dddd", "fQf">;
