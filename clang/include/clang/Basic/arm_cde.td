//===--- arm_cde.td - ACLE intrinsic functions for CDE --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the set of ACLE-specified source-level intrinsic
// functions wrapping the CDE instructions.
//
//===----------------------------------------------------------------------===//

include "arm_mve_defs.td"

// f64 is not defined in arm_mve_defs.td because MVE instructions only work with
// f16 and f32
def f64: PrimitiveType<"f", 64>;

// Float<t> expects t to be a scalar type, and expands to the floating-point
// type of the same width.
class Float<Type t>: ComplexType<(CTO_CopyKind t, f32)>;
def FScalar: Float<Scalar>;

// ACLE CDE intrinsic
class CDEIntrinsic<Type ret, dag args, dag codegen>
  : Intrinsic<ret, args, codegen> {
  let builtinExtension = "cde";
}

// Immediate (in range [0, 2^numBits - 1])
class IB_ConstBits<int numBits> : IB_ConstRange<0, !add(!shl(1, numBits), -1)>;
// numBits-wide immediate of type u32
class CDEImmediateBits<int numBits> : Immediate<u32, IB_ConstBits<numBits>>;

// LLVM IR CDE intrinsic
class CDEIRInt<string name, list<Type> params = [], bit appendKind = 0>
      : IRIntBase<"arm_cde_" # name, params, appendKind>;

// Coprocessor immediate
def imm_coproc : Immediate<sint, IB_ConstRange<0, 7>>;

// Immediate integer parameters
def imm_3b : CDEImmediateBits<3>;
def imm_4b : CDEImmediateBits<4>;
def imm_6b :  CDEImmediateBits<6>;
def imm_7b :  CDEImmediateBits<7>;
def imm_9b :  CDEImmediateBits<9>;
def imm_11b : CDEImmediateBits<11>;
def imm_12b : CDEImmediateBits<12>;
def imm_13b : CDEImmediateBits<13>;

// CX* instructions operating on GPRs
multiclass CDE_CX_m<dag argsImm, dag argsReg, dag cgArgs> {
  defvar cp = (args imm_coproc:$cp);
  let pnt = PNT_None, params = T.None in {
    def "" : CDEIntrinsic<u32, !con(cp, argsReg, argsImm),
                               !con((CDEIRInt<NAME> $cp), cgArgs, (? $imm))>;
    def a  : CDEIntrinsic<u32, !con(cp, (args u32:$acc), argsReg, argsImm),
                               !con((CDEIRInt<NAME # "a"> $cp, $acc),
                                    cgArgs, (? $imm))>;

    def d :
      CDEIntrinsic<u64, !con(cp, argsReg, argsImm),
            (seq !con((CDEIRInt<NAME # "d"> $cp), cgArgs, (? $imm)):$pair,
                 (or (shl (u64 (xval $pair, 1)), (u64 32)),
                          (u64 (xval $pair, 0))))>;
    def da :
      CDEIntrinsic<u64, !con(cp, (args u64:$acc), argsReg, argsImm),
            (seq (u32 (lshr $acc, (u64 32))):$acc_hi,
                 (u32 $acc):$acc_lo,
                 !con((CDEIRInt<NAME # "da"> $cp, $acc_lo, $acc_hi), cgArgs,
                       (? $imm)):$pair,
                 (or (shl (u64 (xval $pair, 1)), (u64 32)),
                          (u64 (xval $pair, 0))))>;
  }
}

defm cx1 : CDE_CX_m<(args imm_13b:$imm), (args), (?)>;
defm cx2 : CDE_CX_m<(args imm_9b:$imm), (args u32:$n), (? $n)>;
defm cx3 : CDE_CX_m<(args imm_6b:$imm), (args u32:$n, u32:$m), (? $n, $m)>;

// VCX* instructions operating on VFP registers
multiclass CDE_VCXFP_m<dag argsImm, dag argsReg32, dag argsReg64, dag cgArgs> {
  defvar cp = (args imm_coproc:$cp);
  let pnt = PNT_None, params = [u32] in {
    def "" : CDEIntrinsic<u32, !con(cp, argsReg32, argsImm),
          (bitcast !con((CDEIRInt<NAME, [f32]> $cp), cgArgs, (? $imm)),
                   Scalar)>;
    def a  : CDEIntrinsic<u32, !con(cp, (args u32:$acc), argsReg32, argsImm),
          (bitcast !con((CDEIRInt<NAME # "a", [f32]> $cp,
                         (bitcast $acc, FScalar)), cgArgs, (? $imm)), Scalar)>;
  }
  let pnt = PNT_None, params = [u64] in {
    def d  : CDEIntrinsic<u64, !con(cp, argsReg64, argsImm),
          (bitcast !con((CDEIRInt<NAME, [f64]> $cp), cgArgs, (? $imm)),
                   Scalar)>;
    def da : CDEIntrinsic<u64, !con(cp, (args u64:$acc), argsReg64, argsImm),
          (bitcast !con((CDEIRInt<NAME # "a", [f64]> $cp,
                         (bitcast $acc, FScalar)), cgArgs, (? $imm)), Scalar)>;
  }
}

defm vcx1: CDE_VCXFP_m<(args imm_11b:$imm), (args), (args), (?)>;
defm vcx2: CDE_VCXFP_m<(args imm_6b:$imm), (args u32:$n), (args u64:$n),
                       (? (bitcast $n, FScalar))>;
defm vcx3: CDE_VCXFP_m<(args imm_3b:$imm),
                       (args u32:$n, u32:$m), (args u64:$n, u64:$m),
                       (? (bitcast $n, FScalar), (bitcast $m, FScalar))>;
