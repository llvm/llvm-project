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

let pnt = PNT_None, params = T.None in
def cx1 : CDEIntrinsic<u32, (args imm_coproc:$cp, imm_13b:$imm),
                            (CDEIRInt<"cx1"> $cp, $imm)>;
