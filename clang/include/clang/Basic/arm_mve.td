//===- arm_mve.td - ACLE intrinsic functions for MVE architecture ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the set of ACLE-specified source-level intrinsic
// functions wrapping the MVE vector instruction set and scalar shift
// operations.
//
// Refer to comments in arm_mve_defs.td for the infrastructure used in
// here, and to MveEmitter.cpp for how those are used in turn to
// generate code.
//
//===----------------------------------------------------------------------===//

include "arm_mve_defs.td"

let params = T.All in
foreach n = [ 2, 4 ] in {
  def "vst"#n#"q": Intrinsic<Void, (args Ptr<Scalar>, MultiVector<n>),
                             (CustomCodegen<"VST24"> n:$NumVectors,
                              "Intrinsic::arm_mve_vst"#n#"q":$IRIntr)>;
  def "vld"#n#"q": Intrinsic<MultiVector<n>, (args CPtr<Scalar>),
                             (CustomCodegen<"VLD24"> n:$NumVectors,
                              "Intrinsic::arm_mve_vld"#n#"q":$IRIntr)>;
}

let params = T.Int in {
def vaddq: Intrinsic<Vector, (args Vector:$a, Vector:$b), (add $a, $b)>;
def vsubq: Intrinsic<Vector, (args Vector:$a, Vector:$b), (sub $a, $b)>;
}

let params = T.Float in {
def vaddqf: Intrinsic<Vector, (args Vector:$a, Vector:$b), (fadd $a, $b)>,
            NameOverride<"vaddq">;
def vsubqf: Intrinsic<Vector, (args Vector:$a, Vector:$b), (fsub $a, $b)>,
            NameOverride<"vsubq">;
}

let params = T.Usual in {
def vaddq_m: Intrinsic<
    Vector, (args Vector:$inactive, Vector:$a, Vector:$b, Predicate:$pred),
    (IRInt<"add_predicated", [Vector, Predicate]> $a, $b, $pred, $inactive)>;
def vsubq_m: Intrinsic<
    Vector, (args Vector:$inactive, Vector:$a, Vector:$b, Predicate:$pred),
    (IRInt<"sub_predicated", [Vector, Predicate]> $a, $b, $pred, $inactive)>;
}

let params = T.Int in {
def vminvq: Intrinsic<Scalar, (args Scalar:$prev, Vector:$vec),
    (Scalar (IRInt<"minv", [Vector], 1> $prev, $vec))>;
def vmaxvq: Intrinsic<Scalar, (args Scalar:$prev, Vector:$vec),
    (Scalar (IRInt<"maxv", [Vector], 1> $prev, $vec))>;
}

foreach half = [ "b", "t" ] in
foreach halfconst = [ !if(!eq(half, "b"), 0, 1) ] in {

let params = [f32], pnt = PNT_None in {

def vcvt#half#q_f16: Intrinsic<
    VecOf<f16>, (args VecOf<f16>:$inactive, Vector:$a),
    (IRInt<"vcvt_narrow"> $inactive, $a, halfconst)>;
def vcvt#half#q_m_f16: Intrinsic<
    VecOf<f16>, (args VecOf<f16>:$inactive, Vector:$a, PredOf<f32>:$pred),
    (IRInt<"vcvt_narrow_predicated"> $inactive, $a, halfconst, $pred)>;

} // params = [f32], pnt = PNT_None

} // loop over half = "b", "t"

let params = T.All32, pnt = PNT_None in
def vldrwq_gather_base_wb: Intrinsic<
    Vector, (args Ptr<VecOf<Unsigned<Scalar>>>:$addr, imm_mem7bit<4>:$offset),
    (seq (IRInt<"vldr_gather_base_wb", [Vector, VecOf<Unsigned<Scalar>>]>
               (load $addr), $offset):$pair,
         (store (xval $pair, 1), $addr),
         (xval $pair, 0))>;

let params = T.All64, pnt = PNT_None in
def vldrdq_gather_base_wb_z: Intrinsic<
    Vector, (args Ptr<VecOf<Unsigned<Scalar>>>:$addr, imm_mem7bit<8>:$offset,
                  Predicate:$pred),
    (seq (IRInt<"vldr_gather_base_wb_predicated", [Vector, VecOf<Unsigned<Scalar>>, Predicate]>
               (load $addr), $offset, $pred):$pair,
         (store (xval $pair, 1), $addr),
         (xval $pair, 0))>;

let params = [Void], pnt = PNT_None in
def urshrl: Intrinsic<u64, (args u64:$value, imm_1to32:$shift),
                      (seq (u32 (lshr $value, (u64 32))):$hi,
                           (u32 $value):$lo,
                           (IRInt<"urshrl"> $lo, $hi, $shift):$pair,
                           (or (shl (u64 (xval $pair, 1)), (u64 32)),
                               (u64 (xval $pair, 0))))>;

let params = T.Int32 in {
def vadcq: Intrinsic<Vector, (args Vector:$a, Vector:$b, Ptr<u32>:$carry),
    (seq (IRInt<"vadc", [Vector]> $a, $b, (shl (load $carry), 29)):$pair,
         (store (and 1, (lshr (xval $pair, 1), 29)), $carry),
         (xval $pair, 0))>;
def vadciq: Intrinsic<Vector, (args Vector:$a, Vector:$b, Ptr<u32>:$carry),
    (seq (IRInt<"vadc", [Vector]> $a, $b, 0):$pair,
         (store (and 1, (lshr (xval $pair, 1), 29)), $carry),
         (xval $pair, 0))>;
def vadcq_m: Intrinsic<Vector, (args Vector:$inactive, Vector:$a, Vector:$b,
                                     Ptr<u32>:$carry, Predicate:$pred),
    (seq (IRInt<"vadc_predicated", [Vector, Predicate]> $inactive, $a, $b,
             (shl (load $carry), 29), $pred):$pair,
         (store (and 1, (lshr (xval $pair, 1), 29)), $carry),
         (xval $pair, 0))>;
def vadciq_m: Intrinsic<Vector, (args Vector:$inactive, Vector:$a, Vector:$b,
                                      Ptr<u32>:$carry, Predicate:$pred),
    (seq (IRInt<"vadc_predicated", [Vector, Predicate]> $inactive, $a, $b,
             0, $pred):$pair,
         (store (and 1, (lshr (xval $pair, 1), 29)), $carry),
         (xval $pair, 0))>;
}
