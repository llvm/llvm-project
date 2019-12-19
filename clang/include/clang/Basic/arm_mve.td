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

let params = T.Usual in
foreach n = [ 2, 4 ] in {
  def "vst"#n#"q": Intrinsic<Void, (args Ptr<Scalar>, MultiVector<n>),
                             (CustomCodegen<"VST24"> n:$NumVectors,
                              "Intrinsic::arm_mve_vst"#n#"q":$IRIntr)>;
  def "vld"#n#"q": Intrinsic<MultiVector<n>, (args CPtr<Scalar>),
                             (CustomCodegen<"VLD24"> n:$NumVectors,
                              "Intrinsic::arm_mve_vld"#n#"q":$IRIntr)>;
}

multiclass bit_op_fp<IRBuilder bitop> {
def "": Intrinsic<Vector, (args Vector:$a, Vector:$b),
   (bitcast (bitop (bitcast $a, UVector), (bitcast $b, UVector)), Vector)>;
}

multiclass bit_op_fp_with_inv<IRBuilder bitop> {
def "": Intrinsic<Vector, (args Vector:$a, Vector:$b),
   (bitcast (bitop (bitcast $a, UVector), (not (bitcast $b, UVector))), Vector)>;
}

let params = T.Signed in {
def vqaddq: Intrinsic<Vector, (args Vector:$a, Vector:$b),
                              (IRIntBase<"sadd_sat", [Vector]> $a, $b)>;
def vqsubq: Intrinsic<Vector, (args Vector:$a, Vector:$b),
                              (IRIntBase<"ssub_sat", [Vector]> $a, $b)>;
}
let params = T.Unsigned in {
def vqaddq_u: Intrinsic<Vector, (args Vector:$a, Vector:$b),
                                (IRIntBase<"uadd_sat", [Vector]> $a, $b)>,
              NameOverride<"vqaddq">;
def vqsubq_u: Intrinsic<Vector, (args Vector:$a, Vector:$b),
                                (IRIntBase<"usub_sat", [Vector]> $a, $b)>,
              NameOverride<"vqsubq">;
}

// Some intrinsics below are implemented not as IR fragments, but as
// special-purpose IR intrinsics. This is because such a general form
// (such as NEON uses) required a variable-width vector size, and we are
// restricted to 128 bit. Although we can possibly get clever with lane
// operations, the consequent IR representation would be very hard to
// write sensibly. In particular, doubling a vector's width would be a
// mess. Other intrinsics just don't translate nicely into IR.
let params = T.Int in {
def vaddq: Intrinsic<Vector, (args Vector:$a, Vector:$b), (add $a, $b)>;
def vhaddq: Intrinsic<Vector, (args Vector:$a, Vector:$b),
                              (IRInt<"vhadd", [Vector]> $a, $b)>;
def vrhaddq: Intrinsic<Vector, (args Vector:$a, Vector:$b),
                               (IRInt<"vrhadd", [Vector]> $a, $b)>;
def vandq: Intrinsic<Vector, (args Vector:$a, Vector:$b), (and $a, $b)>;
def vbicq: Intrinsic<Vector, (args Vector:$a, Vector:$b), (and $a, (not $b))>;
def veorq: Intrinsic<Vector, (args Vector:$a, Vector:$b), (xor $a, $b)>;
def vornq: Intrinsic<Vector, (args Vector:$a, Vector:$b), (or $a, (not $b))>;
def vorrq: Intrinsic<Vector, (args Vector:$a, Vector:$b), (or $a, $b)>;
def vsubq: Intrinsic<Vector, (args Vector:$a, Vector:$b), (sub $a, $b)>;
def vhsubq: Intrinsic<Vector, (args Vector:$a, Vector:$b),
                              (IRInt<"vhsub", [Vector]> $a, $b)>;
def vmulq: Intrinsic<Vector, (args Vector:$a, Vector:$b), (mul $a, $b)>;
def vmulhq: Intrinsic<Vector, (args Vector:$a, Vector:$b),
                              (IRInt<"vmulh", [Vector]> $a, $b)>;
def vrmulhq: Intrinsic<Vector, (args Vector:$a, Vector:$b),
                               (IRInt<"vrmulh", [Vector]> $a, $b)>;
def vmullbq_int: Intrinsic<DblVector, (args Vector:$a, Vector:$b),
                                      (IRInt<"vmull", [DblVector, Vector]>
                                       $a, $b, 0)>;
def vmulltq_int: Intrinsic<DblVector, (args Vector:$a, Vector:$b),
                                      (IRInt<"vmull", [DblVector, Vector]>
                                       $a, $b, 1)>;
}
let params = T.Signed in {
def vqdmulhq: Intrinsic<Vector, (args Vector:$a, Vector:$b),
                                (IRInt<"vqdmulh", [Vector]> $a, $b)>;
def vqrdmulhq: Intrinsic<Vector, (args Vector:$a, Vector:$b),
                                 (IRInt<"vqrdmulh", [Vector]> $a, $b)>;
}

let params = T.Poly, overrideKindLetter = "p" in {
def vmullbq_poly: Intrinsic<DblVector, (args Vector:$a, Vector:$b),
                                       (IRInt<"vmull_poly", [DblVector, Vector]>
                                        $a, $b, 0)>;
def vmulltq_poly: Intrinsic<DblVector, (args Vector:$a, Vector:$b),
                                       (IRInt<"vmull_poly", [DblVector, Vector]>
                                        $a, $b, 1)>;
}

let params = T.Float in {
def vaddqf: Intrinsic<Vector, (args Vector:$a, Vector:$b), (fadd $a, $b)>,
            NameOverride<"vaddq">;
defm vandqf: bit_op_fp<and>, NameOverride<"vandq">;
defm vbicqf: bit_op_fp_with_inv<and>, NameOverride<"vbicq">;
defm veorqf: bit_op_fp<xor>, NameOverride<"veorq">;
defm vornqf: bit_op_fp_with_inv<or>, NameOverride<"vornq">;
defm vorrqf: bit_op_fp<or>, NameOverride<"vorrq">;
def vsubqf: Intrinsic<Vector, (args Vector:$a, Vector:$b), (fsub $a, $b)>,
            NameOverride<"vsubq">;
def vmulqf: Intrinsic<Vector, (args Vector:$a, Vector:$b), (fmul $a, $b)>,
            NameOverride<"vmulq">;
}

// The bitcasting below is not overcomplicating the IR because while
// Vector and UVector may be different vector types at the C level i.e.
// vectors of same size signed/unsigned ints. Once they're lowered
// to IR, they are just bit vectors with no sign at all, so the
// bitcasts will be automatically elided by IRBuilder.
multiclass predicated_bit_op_fp<string int_op> {
def "": Intrinsic<Vector, (args Vector:$inactive, Vector:$a, Vector:$b,
                                Predicate:$pred),
    (bitcast (IRInt<int_op, [UVector, Predicate]>
                    (bitcast $a, UVector),
                    (bitcast $b, UVector),
                    $pred,
                    (bitcast $inactive, UVector)), Vector)>;
}

// Plain intrinsics
let params = T.Usual in {
def vabdq: Intrinsic<Vector, (args Vector:$a, Vector:$b),
                             (IRInt<"vabd", [Vector]> $a, $b)>;
}

multiclass VectorVectorArithmetic<string operation, int wantXVariant = 1> {
  defm "" : IntrinsicMX<Vector, (args Vector:$a, Vector:$b,
                                 Predicate:$pred),
                                (IRInt<operation, [Vector, Predicate]> $a, $b,
                                 $pred, $inactive),
                                wantXVariant>;
}

multiclass VectorVectorArithmeticBitcast<string operation> {
  defm "" : IntrinsicMX<Vector, (args Vector:$a, Vector:$b,
                                 Predicate:$pred),
                                (bitcast (IRInt<operation, [UVector, Predicate]>
                                          (bitcast $a, UVector),
                                          (bitcast $b, UVector),
                                          $pred,
                                          (bitcast $inactive, UVector)), Vector)>;
}

// Predicated intrinsics
let params = T.Usual in {
  defm vabdq : VectorVectorArithmetic<"abd_predicated">;
  defm vaddq : VectorVectorArithmetic<"add_predicated">;
  defm vsubq : VectorVectorArithmetic<"sub_predicated">;
  defm vmulq : VectorVectorArithmetic<"mul_predicated">;
  defm vandq : VectorVectorArithmeticBitcast<"and_predicated">;
  defm vbicq : VectorVectorArithmeticBitcast<"bic_predicated">;
  defm veorq : VectorVectorArithmeticBitcast<"eor_predicated">;
  defm vornq : VectorVectorArithmeticBitcast<"orn_predicated">;
  defm vorrq : VectorVectorArithmeticBitcast<"orr_predicated">;
}

multiclass DblVectorVectorArithmetic<string operation, dag top> {
  defm "" : IntrinsicMX<DblVector, (args Vector:$a, Vector:$b,
                                    Predicate:$pred),
                                   (IRInt<operation,
                                          [DblVector, Vector, Predicate]>
                                    $a, $b, top, $pred, $inactive)>;
}

// Predicated intrinsics - Int types only
let params = T.Int in {
  defm vminq : VectorVectorArithmetic<"min_predicated">;
  defm vmaxq : VectorVectorArithmetic<"max_predicated">;
  defm vmulhq : VectorVectorArithmetic<"mulh_predicated">;
  defm vrmulhq : VectorVectorArithmetic<"rmulh_predicated">;
  defm vqaddq : VectorVectorArithmetic<"qadd_predicated", 0>;
  defm vhaddq : VectorVectorArithmetic<"hadd_predicated">;
  defm vrhaddq : VectorVectorArithmetic<"rhadd_predicated">;
  defm vqsubq : VectorVectorArithmetic<"qsub_predicated", 0>;
  defm vhsubq : VectorVectorArithmetic<"hsub_predicated">;
  defm vmullbq_int : DblVectorVectorArithmetic<"mull_int_predicated", (u32 0)>;
  defm vmulltq_int : DblVectorVectorArithmetic<"mull_int_predicated", (u32 1)>;
}
let params = T.Signed in {
  defm vqdmulhq : VectorVectorArithmetic<"qdmulh_predicated", 0>;
  defm vqrdmulhq : VectorVectorArithmetic<"qrdmulh_predicated", 0>;
}

let params = T.Poly, overrideKindLetter = "p" in {
  defm vmullbq_poly : DblVectorVectorArithmetic<"mull_poly_predicated", (u32 0)>;
  defm vmulltq_poly : DblVectorVectorArithmetic<"mull_poly_predicated", (u32 1)>;
}

// Predicated intrinsics - Float types only
let params = T.Float in {
  defm vminnmq : VectorVectorArithmetic<"min_predicated">;
  defm vmaxnmq : VectorVectorArithmetic<"max_predicated">;
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

multiclass compare_with_pred<string condname, dag arguments,
                             dag cmp, string suffix> {
  // Make the predicated and unpredicated versions of a single comparison.
  def: Intrinsic<Predicate, arguments,
                 (u16 (IRInt<"pred_v2i", [Predicate]> cmp))>,
       NameOverride<"vcmp" # condname # "q" # suffix>;
  def: Intrinsic<Predicate, !con(arguments, (args Predicate:$inpred)),
                 (u16 (IRInt<"pred_v2i", [Predicate]> (and $inpred, cmp)))>,
       NameOverride<"vcmp" # condname # "q_m" # suffix>;
}

multiclass compare<string condname, IRBuilder cmpop> {
  // Make all four variants of a comparison: the vector/vector and
  // vector/scalar forms, each using compare_with_pred to make a
  // predicated and unpredicated version.
  defm: compare_with_pred<condname, (args Vector:$va, Vector:$vb),
                          (cmpop $va, $vb), "">;
  let pnt = PNT_NType in {
    defm: compare_with_pred<condname, (args Vector:$va, unpromoted<Scalar>:$sb),
                            (cmpop $va, (splat $sb)), "_n">;
  }
}
let params = T.Int in {
  defm: compare<"eq", icmp_eq>;
  defm: compare<"ne", icmp_ne>;
}
let params = T.Signed in {
  defm: compare<"gt", icmp_sgt>;
  defm: compare<"ge", icmp_sge>;
  defm: compare<"lt", icmp_slt>;
  defm: compare<"le", icmp_sle>;
}
let params = T.Unsigned in {
  defm: compare<"hi", icmp_ugt>;
  defm: compare<"cs", icmp_uge>;
}
let params = T.Float in {
  defm: compare<"eq", fcmp_eq>;
  defm: compare<"ne", fcmp_ne>;
  defm: compare<"gt", fcmp_gt>;
  defm: compare<"ge", fcmp_ge>;
  defm: compare<"lt", fcmp_lt>;
  defm: compare<"le", fcmp_le>;
}

let params = T.Signed in {
  def vminq: Intrinsic<Vector, (args Vector:$a, Vector:$b),
                               (select (icmp_sle $a, $b), $a, $b)>;
  def vmaxq: Intrinsic<Vector, (args Vector:$a, Vector:$b),
                               (select (icmp_sge $a, $b), $a, $b)>;
}
let params = T.Unsigned in {
  def vminqu: Intrinsic<Vector, (args Vector:$a, Vector:$b),
                                (select (icmp_ule $a, $b), $a, $b)>,
              NameOverride<"vminq">;
  def vmaxqu: Intrinsic<Vector, (args Vector:$a, Vector:$b),
                                (select (icmp_uge $a, $b), $a, $b)>,
              NameOverride<"vmaxq">;
}
let params = T.Float in {
  def vminnmq: Intrinsic<Vector, (args Vector:$a, Vector:$b),
                               (IRIntBase<"minnum", [Vector]> $a, $b)>;
  def vmaxnmq: Intrinsic<Vector, (args Vector:$a, Vector:$b),
                               (IRIntBase<"maxnum", [Vector]> $a, $b)>;
}

def vpselq: Intrinsic<Vector, (args Vector:$t, Vector:$f, Predicate:$pred),
                      (select $pred, $t, $f)> { let params = T.Usual; }
def vpselq_64: Intrinsic<
    Vector, (args Vector:$t, Vector:$f, PredOf<u32>:$pred),
            (bitcast (select $pred, (bitcast $t, VecOf<u32>),
                                    (bitcast $f, VecOf<u32>)), Vector)>,
    NameOverride<"vpselq"> { let params = T.All64; }

let params = [Void], pnt = PNT_None in {

  multiclass vctp<Type pred, string intname> {
    def "": Intrinsic<pred, (args u32:$val),
        (u16 (IRInt<"pred_v2i", [pred]> (IRIntBase<intname> $val)))>;
    def _m: Intrinsic<pred, (args u32:$val, pred:$inpred),
        (u16 (IRInt<"pred_v2i", [pred]> (and $inpred,
                                         (IRIntBase<intname> $val))))>;
  }
  defm vctp8q:  vctp<PredOf<u8>,  "arm_mve_vctp8">;
  defm vctp16q: vctp<PredOf<u16>, "arm_mve_vctp16">;
  defm vctp32q: vctp<PredOf<u32>, "arm_mve_vctp32">;
  defm vctp64q: vctp<PredOf<u64>, "arm_mve_vctp64">;

  def vpnot: Intrinsic<PredOf<u8>, (args unpromoted<PredOf<u8>>:$pred),
                       (xor $pred, (u16 65535))>;

}

multiclass contiguous_load<string mnemonic, PrimitiveType memtype,
                           list<Type> same_size, list<Type> wider> {
  // Intrinsics named with explicit memory and element sizes that match:
  // vldrbq_?8, vldrhq_?16, vldrwq_?32.
  let params = same_size, pnt = PNT_None in {
    def: Intrinsic<Vector, (args CPtr<CopyKind<same_size[0], Scalar>>:$addr),
                   (load (address (CPtr<Vector> $addr), !srl(memtype.size,3)))>,
         NameOverride<mnemonic>;
    def: Intrinsic<Vector, (args CPtr<CopyKind<same_size[0], Scalar>>:$addr,
                                 Predicate:$pred),
                   (IRIntBase<"masked_load", [Vector, CPtr<Vector>]>
                        (CPtr<Vector> $addr), !srl(memtype.size,3),
                        $pred, (zeroinit Vector))>,
         NameOverride<mnemonic # "_z">;
  }

  // Synonyms for the above, with the generic name vld1q that just means
  // 'memory and element sizes match', and allows convenient polymorphism with
  // the memory and element types covariant.
  let params = same_size in {
    def: Intrinsic<Vector, (args CPtr<CopyKind<same_size[0], Scalar>>:$addr),
                   (load (address (CPtr<Vector> $addr), !srl(memtype.size,3)))>,
         NameOverride<"vld1q">;
    def: Intrinsic<Vector, (args CPtr<CopyKind<same_size[0], Scalar>>:$addr,
                                 Predicate:$pred),
                   (IRIntBase<"masked_load", [Vector, CPtr<Vector>]>
                        (CPtr<Vector> $addr), !srl(memtype.size,3),
                        $pred, (zeroinit Vector))>,
         NameOverride<"vld1q_z">;
  }

  // Intrinsics with the memory size narrower than the vector element, so that
  // they load less than 128 bits of memory and sign/zero extend each loaded
  // value into a wider vector lane.
  let params = wider, pnt = PNT_None in {
    def: Intrinsic<Vector, (args CPtr<CopyKind<same_size[0], Scalar>>:$addr),
                   (extend (load (address (CPtr<NarrowedVecOf<memtype,Vector>>
                                           $addr), !srl(memtype.size,3))),
                           Vector, (unsignedflag Scalar))>,
         NameOverride<mnemonic>;
    def: Intrinsic<Vector, (args CPtr<CopyKind<same_size[0], Scalar>>:$addr,
                                 Predicate:$pred),
                   (extend (IRIntBase<"masked_load",
                                      [NarrowedVecOf<memtype,Vector>,
                                      CPtr<NarrowedVecOf<memtype,Vector>>]>
                                (CPtr<NarrowedVecOf<memtype,Vector>> $addr),
                                !srl(memtype.size,3), $pred,
                                (zeroinit NarrowedVecOf<memtype,Vector>)),
                           Vector, (unsignedflag Scalar))>,
         NameOverride<mnemonic # "_z">;
  }
}

defm: contiguous_load<"vldrbq", u8, T.All8, !listconcat(T.Int16, T.Int32)>;
defm: contiguous_load<"vldrhq", u16, T.All16, T.Int32>;
defm: contiguous_load<"vldrwq", u32, T.All32, []>;

multiclass contiguous_store<string mnemonic, PrimitiveType memtype,
                           list<Type> same_size, list<Type> wider> {
  // Intrinsics named with explicit memory and element sizes that match:
  // vstrbq_?8, vstrhq_?16, vstrwq_?32.
  let params = same_size in {
    def: Intrinsic<Void, (args Ptr<CopyKind<same_size[0], Scalar>>:$addr,
                               Vector:$value),
                   (store $value,
                          (address (Ptr<Vector> $addr), !srl(memtype.size,3)))>,
         NameOverride<mnemonic>;
    def: Intrinsic<Void, (args Ptr<CopyKind<same_size[0], Scalar>>:$addr,
                               Vector:$value, Predicate:$pred),
                   (IRIntBase<"masked_store", [Vector, Ptr<Vector>]>
                        $value, (Ptr<Vector> $addr),
                        !srl(memtype.size,3), $pred)>,
         NameOverride<mnemonic # "_p">;
  }

  // Synonyms for the above, with the generic name vst1q that just means
  // 'memory and element sizes match', and allows convenient polymorphism with
  // the memory and element types covariant.
  let params = same_size in {
    def: Intrinsic<Void, (args Ptr<CopyKind<same_size[0], Scalar>>:$addr,
                               Vector:$value),
                   (store $value,
                          (address (Ptr<Vector> $addr), !srl(memtype.size,3)))>,
         NameOverride<"vst1q">;
    def: Intrinsic<Void, (args Ptr<CopyKind<same_size[0], Scalar>>:$addr,
                               Vector:$value, Predicate:$pred),
                   (IRIntBase<"masked_store", [Vector, Ptr<Vector>]>
                        $value, (Ptr<Vector> $addr),
                        !srl(memtype.size,3), $pred)>,
         NameOverride<"vst1q_p">;
  }

  // Intrinsics with the memory size narrower than the vector element, so that
  // they store less than 128 bits of memory, truncating each vector lane into
  // a narrower value to store.
  let params = wider in {
    def: Intrinsic<Void, (args Ptr<CopyKind<same_size[0], Scalar>>:$addr,
                               Vector:$value),
                   (store (trunc $value, NarrowedVecOf<memtype,Vector>),
                          (address (Ptr<NarrowedVecOf<memtype,Vector>> $addr),
                                   !srl(memtype.size,3)))>,
         NameOverride<mnemonic>;
    def: Intrinsic<Void, (args Ptr<CopyKind<same_size[0], Scalar>>:$addr,
                               Vector:$value, Predicate:$pred),
                   (IRIntBase<"masked_store",
                              [NarrowedVecOf<memtype,Vector>,
                               Ptr<NarrowedVecOf<memtype,Vector>>]>
                        (trunc $value, NarrowedVecOf<memtype,Vector>),
                        (Ptr<NarrowedVecOf<memtype,Vector>> $addr),
                        !srl(memtype.size,3), $pred)>,
         NameOverride<mnemonic # "_p">;
  }
}

defm: contiguous_store<"vstrbq", u8, T.All8, !listconcat(T.Int16, T.Int32)>;
defm: contiguous_store<"vstrhq", u16, T.All16, T.Int32>;
defm: contiguous_store<"vstrwq", u32, T.All32, []>;

multiclass gather_base<list<Type> types, int size> {
  let params = types, pnt = PNT_None in {
    def _gather_base: Intrinsic<
      Vector, (args UVector:$addr, imm_mem7bit<size>:$offset),
      (IRInt<"vldr_gather_base", [Vector, UVector]> $addr, $offset)>;

    def _gather_base_z: Intrinsic<
      Vector, (args UVector:$addr, imm_mem7bit<size>:$offset, Predicate:$pred),
      (IRInt<"vldr_gather_base_predicated", [Vector, UVector, Predicate]>
          $addr, $offset, $pred)>;

    def _gather_base_wb: Intrinsic<
      Vector, (args Ptr<UVector>:$addr, imm_mem7bit<size>:$offset),
      (seq (IRInt<"vldr_gather_base_wb", [Vector, UVector]>
               (load $addr), $offset):$pair,
           (store (xval $pair, 1), $addr),
           (xval $pair, 0))>;

    def _gather_base_wb_z: Intrinsic<
      Vector, (args Ptr<UVector>:$addr, imm_mem7bit<size>:$offset,
                    Predicate:$pred),
      (seq (IRInt<"vldr_gather_base_wb_predicated",
                  [Vector, UVector, Predicate]>
               (load $addr), $offset, $pred):$pair,
           (store (xval $pair, 1), $addr),
           (xval $pair, 0))>;
  }
}

defm vldrwq: gather_base<T.All32, 4>;
defm vldrdq: gather_base<T.All64, 8>;

multiclass scatter_base<list<Type> types, int size> {
  let params = types in {
    def _scatter_base: Intrinsic<
      Void, (args UVector:$addr, imm_mem7bit<size>:$offset, Vector:$data),
      (IRInt<"vstr_scatter_base", [UVector, Vector]> $addr, $offset, $data)>;

    def _scatter_base_p: Intrinsic<
      Void, (args UVector:$addr, imm_mem7bit<size>:$offset, Vector:$data,
                  Predicate:$pred),
      (IRInt<"vstr_scatter_base_predicated", [UVector, Vector, Predicate]>
          $addr, $offset, $data, $pred)>;

    def _scatter_base_wb: Intrinsic<
      Void, (args Ptr<UVector>:$addr, imm_mem7bit<size>:$offset, Vector:$data),
      (seq (IRInt<"vstr_scatter_base_wb", [UVector, Vector]>
                 (load $addr), $offset, $data):$wbaddr,
           (store $wbaddr, $addr))>;

    def _scatter_base_wb_p: Intrinsic<
      Void, (args Ptr<UVector>:$addr, imm_mem7bit<size>:$offset,
                    Vector:$data, Predicate:$pred),
      (seq (IRInt<"vstr_scatter_base_wb_predicated",
                  [UVector, Vector, Predicate]>
               (load $addr), $offset, $data, $pred):$wbaddr,
           (store $wbaddr, $addr))>;
  }
}

defm vstrwq: scatter_base<T.All32, 4>;
defm vstrdq: scatter_base<T.All64, 8>;

multiclass gather_offset_unshifted<list<Type> types, PrimitiveType memtype> {
  let params = types in {
    def _gather_offset: Intrinsic<
      Vector, (args CPtr<CopyKind<memtype, Scalar>>:$base, UVector:$offsets),
      (IRInt<"vldr_gather_offset",
             [Vector, CPtr<CopyKind<memtype, Scalar>>, UVector]>
          $base, $offsets, memtype.size, 0, (unsignedflag Scalar))>;
    def _gather_offset_z: Intrinsic<
      Vector, (args CPtr<CopyKind<memtype, Scalar>>:$base, UVector:$offsets,
                    Predicate:$pred),
      (IRInt<"vldr_gather_offset_predicated",
             [Vector, CPtr<CopyKind<memtype, Scalar>>, UVector, Predicate]>
          $base, $offsets, memtype.size, 0, (unsignedflag Scalar), $pred)>;
  }
}

multiclass gather_offset_shifted<list<Type> types, PrimitiveType memtype,
                                 int shift> {
  let params = types in {
    def _gather_shifted_offset: Intrinsic<
      Vector, (args CPtr<CopyKind<memtype, Scalar>>:$base, UVector:$offsets),
      (IRInt<"vldr_gather_offset",
             [Vector, CPtr<CopyKind<memtype, Scalar>>, UVector]>
          $base, $offsets, memtype.size, shift, (unsignedflag Scalar))>;
    def _gather_shifted_offset_z: Intrinsic<
      Vector, (args CPtr<CopyKind<memtype, Scalar>>:$base, UVector:$offsets,
                    Predicate:$pred),
      (IRInt<"vldr_gather_offset_predicated",
             [Vector, CPtr<CopyKind<memtype, Scalar>>, UVector, Predicate]>
          $base, $offsets, memtype.size, shift, (unsignedflag Scalar), $pred)>;
  }
}

multiclass gather_offset_both<list<Type> types, PrimitiveType memtype,
                              int shift> {
  defm "": gather_offset_unshifted<types, memtype>;
  defm "": gather_offset_shifted<types, memtype, shift>;
}

defm vldrbq: gather_offset_unshifted<!listconcat(T.All8, T.Int16, T.Int32), u8>;
defm vldrhq: gather_offset_both<!listconcat(T.All16, T.Int32), u16, 1>;
defm vldrwq: gather_offset_both<T.All32, u32, 2>;
defm vldrdq: gather_offset_both<T.Int64, u64, 3>;

multiclass scatter_offset_unshifted<list<Type> types, PrimitiveType memtype> {
  let params = types in {
    def _scatter_offset: Intrinsic<
      Void, (args Ptr<CopyKind<memtype, Scalar>>:$base, UVector:$offsets,
                  Vector:$data),
      (IRInt<"vstr_scatter_offset",
             [Ptr<CopyKind<memtype, Scalar>>, UVector, Vector]>
          $base, $offsets, $data, memtype.size, 0)>;
    def _scatter_offset_p: Intrinsic<
      Void, (args Ptr<CopyKind<memtype, Scalar>>:$base, UVector:$offsets,
                  Vector:$data, Predicate:$pred),
      (IRInt<"vstr_scatter_offset_predicated",
             [Ptr<CopyKind<memtype, Scalar>>, UVector, Vector, Predicate]>
          $base, $offsets, $data, memtype.size, 0, $pred)>;
  }
}

multiclass scatter_offset_shifted<list<Type> types, PrimitiveType memtype,
                                  int shift> {
  let params = types in {
    def _scatter_shifted_offset: Intrinsic<
      Void, (args Ptr<CopyKind<memtype, Scalar>>:$base, UVector:$offsets,
                  Vector:$data),
      (IRInt<"vstr_scatter_offset",
             [Ptr<CopyKind<memtype, Scalar>>, UVector, Vector]>
          $base, $offsets, $data, memtype.size, shift)>;
    def _scatter_shifted_offset_p: Intrinsic<
      Void, (args Ptr<CopyKind<memtype, Scalar>>:$base, UVector:$offsets,
                  Vector:$data, Predicate:$pred),
      (IRInt<"vstr_scatter_offset_predicated",
             [Ptr<CopyKind<memtype, Scalar>>, UVector, Vector, Predicate]>
          $base, $offsets, $data, memtype.size, shift, $pred)>;
  }
}

multiclass scatter_offset_both<list<Type> types, PrimitiveType memtype,
                               int shift> {
  defm "": scatter_offset_unshifted<types, memtype>;
  defm "": scatter_offset_shifted<types, memtype, shift>;
}

defm vstrbq: scatter_offset_unshifted<!listconcat(T.All8,T.Int16,T.Int32), u8>;
defm vstrhq: scatter_offset_both<!listconcat(T.All16, T.Int32), u16, 1>;
defm vstrwq: scatter_offset_both<T.All32, u32, 2>;
defm vstrdq: scatter_offset_both<T.Int64, u64, 3>;

let params = T.Int in {
  def vshlq_n: Intrinsic<Vector, (args Vector:$v, imm_0toNm1:$sh),
                         (shl $v, (splat (Scalar $sh)))>;
  defm vshlq: IntrinsicMX<Vector, (args Vector:$v, imm_0toNm1:$sh,
                                        Predicate:$pred),
      (IRInt<"shl_imm_predicated", [Vector, Predicate]>
           $v, $sh, $pred, $inactive), 1, "_n">;

  let pnt = PNT_NType in {
    def vshrq_n: Intrinsic<Vector, (args Vector:$v, imm_1toN:$sh),
                            (immshr $v, $sh, (unsignedflag Scalar))>;
    defm vshrq: IntrinsicMX<Vector, (args Vector:$v, imm_1toN:$sh,
                                          Predicate:$pred),
        (IRInt<"shr_imm_predicated", [Vector, Predicate]>
             $v, $sh, (unsignedflag Scalar), $pred, $inactive), 1, "_n">;
  }
}

let params = T.Int in {
  def vqshlq_n: Intrinsic<Vector, (args Vector:$v, imm_0toNm1:$sh),
       (IRInt<"vqshl_imm", [Vector]> $v, $sh, (unsignedflag Scalar))>;
  def vqshlq_m_n: Intrinsic<Vector, (args Vector:$inactive, Vector:$v,
                                          imm_0toNm1:$sh, Predicate:$pred),
       (IRInt<"vqshl_imm_predicated", [Vector, Predicate]>
            $v, $sh, (unsignedflag Scalar), $pred, $inactive)>;

  let pnt = PNT_NType in {
    def vrshrq_n: Intrinsic<Vector, (args Vector:$v, imm_1toN:$sh),
         (IRInt<"vrshr_imm", [Vector]> $v, $sh, (unsignedflag Scalar))>;
    defm vrshrq: IntrinsicMX<Vector, (args Vector:$v, imm_1toN:$sh,
                                           Predicate:$pred),
         (IRInt<"vrshr_imm_predicated", [Vector, Predicate]>
              $v, $sh, (unsignedflag Scalar), $pred, $inactive), 1, "_n">;
  }
}

let params = T.Signed, pnt = PNT_NType in {
  def vqshluq_n: Intrinsic<UVector, (args Vector:$v, imm_0toNm1:$sh),
       (IRInt<"vqshlu_imm", [Vector]> $v, $sh)>;
  def vqshluq_m_n: Intrinsic<UVector, (args UVector:$inactive, Vector:$v,
                                            imm_0toNm1:$sh, Predicate:$pred),
       (IRInt<"vqshlu_imm_predicated", [Vector, Predicate]>
            $v, $sh, $pred, $inactive)>;
}

multiclass vshll_imm<int top> {
  let params = !listconcat(T.Int8, T.Int16), pnt = PNT_NType in {
    def _n: Intrinsic<DblVector, (args Vector:$v, imm_1toN:$sh),
        (IRInt<"vshll_imm", [DblVector, Vector]>
            $v, $sh, (unsignedflag Scalar), top)>;
    defm "": IntrinsicMX<DblVector, (args Vector:$v, imm_1toN:$sh,
                                          Predicate:$pred),
        (IRInt<"vshll_imm_predicated", [DblVector, Vector, Predicate]>
            $v, $sh, (unsignedflag Scalar), top, $pred, $inactive), 1, "_n">;
  }
}
defm vshllbq : vshll_imm<0>;
defm vshlltq : vshll_imm<1>;

// Base class for the scalar shift intrinsics.
class ScalarShift<Type argtype, dag shiftCountArg, dag shiftCodeGen>:
  Intrinsic<argtype, !con((args argtype:$value), shiftCountArg), shiftCodeGen> {
  let params = [Void];
  let pnt = PNT_None;
}

// Subclass that includes the machinery to take a 64-bit input apart
// into halves, retrieve the two halves of a shifted output as a pair,
// and glue the pieces of the pair back into an i64 for output.
class LongScalarShift<Type argtype, dag shiftCountArg, dag shiftCodeGen>:
   ScalarShift<argtype, shiftCountArg,
               (seq (u32 (lshr $value, (argtype 32))):$hi,
                    (u32 $value):$lo,
                    shiftCodeGen:$pair,
                    (or (shl (u64 (xval $pair, 1)), (u64 32)),
                             (u64 (xval $pair, 0))))>;

// The family of saturating/rounding scalar shifts that take an
// immediate shift count. They come in matched 32- and 64-bit pairs.
multiclass ScalarSaturatingShiftImm<Type arg32, Type arg64> {
  def "": ScalarShift<arg32, (args imm_1to32:$sh),
                      (IRInt<NAME> $value, $sh)>;
  def l:  LongScalarShift<arg64, (args imm_1to32:$sh),
                          (IRInt<NAME # "l"> $lo, $hi, $sh)>;
}
defm uqshl: ScalarSaturatingShiftImm<u32, u64>;
defm urshr: ScalarSaturatingShiftImm<u32, u64>;
defm sqshl: ScalarSaturatingShiftImm<s32, s64>;
defm srshr: ScalarSaturatingShiftImm<s32, s64>;

// The family of saturating/rounding scalar shifts that take a
// register shift count. They also have 32- and 64-bit forms, but the
// 64-bit form also has a version that saturates to 48 bits, so the IR
// intrinsic takes an extra saturation-type operand.
multiclass ScalarSaturatingShiftReg<Type arg32, Type arg64> {
  def "":          ScalarShift<arg32, (args s32:$sh),
                               (IRInt<NAME> $value, $sh)>;
  def l:       LongScalarShift<arg64, (args s32:$sh),
                               (IRInt<NAME # "l"> $lo, $hi, $sh, 64)>;
  def l_sat48: LongScalarShift<arg64, (args s32:$sh),
                               (IRInt<NAME # "l"> $lo, $hi, $sh, 48)>;
}
defm uqrshl: ScalarSaturatingShiftReg<u32, u64>;
defm sqrshr: ScalarSaturatingShiftReg<s32, s64>;

// The intrinsics for LSLL and ASRL come in 64-bit versions only, with
// no saturation count.
def lsll: LongScalarShift<u64, (args s32:$sh), (IRInt<"lsll"> $lo, $hi, $sh)>;
def asrl: LongScalarShift<s64, (args s32:$sh), (IRInt<"asrl"> $lo, $hi, $sh)>;

let params = T.Int32 in {
def vadcq: Intrinsic<Vector, (args Vector:$a, Vector:$b, Ptr<uint>:$carry),
    (seq (IRInt<"vadc", [Vector]> $a, $b, (shl (load $carry), 29)):$pair,
         (store (and 1, (lshr (xval $pair, 1), 29)), $carry),
         (xval $pair, 0))>;
def vadciq: Intrinsic<Vector, (args Vector:$a, Vector:$b, Ptr<uint>:$carry),
    (seq (IRInt<"vadc", [Vector]> $a, $b, 0):$pair,
         (store (and 1, (lshr (xval $pair, 1), 29)), $carry),
         (xval $pair, 0))>;
def vadcq_m: Intrinsic<Vector, (args Vector:$inactive, Vector:$a, Vector:$b,
                                     Ptr<uint>:$carry, Predicate:$pred),
    (seq (IRInt<"vadc_predicated", [Vector, Predicate]> $inactive, $a, $b,
             (shl (load $carry), 29), $pred):$pair,
         (store (and 1, (lshr (xval $pair, 1), 29)), $carry),
         (xval $pair, 0))>;
def vadciq_m: Intrinsic<Vector, (args Vector:$inactive, Vector:$a, Vector:$b,
                                      Ptr<uint>:$carry, Predicate:$pred),
    (seq (IRInt<"vadc_predicated", [Vector, Predicate]> $inactive, $a, $b,
             0, $pred):$pair,
         (store (and 1, (lshr (xval $pair, 1), 29)), $carry),
         (xval $pair, 0))>;
}

multiclass VectorComplexAddPred<dag not_halving, dag angle> {
  def "" : Intrinsic<Vector, (args Vector:$a, Vector:$b),
     (IRInt<"vcaddq", [Vector]> not_halving, angle, $a, $b)>;
  defm "" : IntrinsicMX<Vector, (args Vector:$a, Vector:$b, Predicate:$pred),
     (IRInt<"vcaddq_predicated", [Vector, Predicate]>
       not_halving, angle, $inactive, $a, $b, $pred)>;
}

multiclass VectorComplexMulPred<dag angle> {
  def "" : Intrinsic<Vector, (args Vector:$a, Vector:$b),
    (IRInt<"vcmulq", [Vector]> angle, $a, $b)>;
  defm "" : IntrinsicMX<Vector, (args Vector:$a, Vector:$b, Predicate:$pred),
    (IRInt<"vcmulq_predicated", [Vector, Predicate]> angle, $inactive, $a, $b,
      $pred)>;
}

multiclass VectorComplexMLAPred<dag angle> {
  def "" : Intrinsic<Vector, (args Vector:$a, Vector:$b, Vector:$c),
    (IRInt<"vcmlaq", [Vector]> angle, $a, $b, $c)>;
  def _m : Intrinsic<Vector, (args Vector:$a, Vector:$b, Vector:$c,
                                   Predicate:$pred),
    (IRInt<"vcmlaq_predicated", [Vector, Predicate]> angle, $a, $b, $c, $pred)>;
}

multiclass VectorComplexAddAngle<dag not_halving> {
  defm _rot90 : VectorComplexAddPred<not_halving, (u32 0)>;
  defm _rot270 : VectorComplexAddPred<not_halving, (u32 1)>;
}

multiclass VectorComplexMulAngle {
  defm ""      : VectorComplexMulPred<(u32 0)>;
  defm _rot90  : VectorComplexMulPred<(u32 1)>;
  defm _rot180 : VectorComplexMulPred<(u32 2)>;
  defm _rot270 : VectorComplexMulPred<(u32 3)>;
}

multiclass VectorComplexMLAAngle {
  defm ""      : VectorComplexMLAPred<(u32 0)>;
  defm _rot90  : VectorComplexMLAPred<(u32 1)>;
  defm _rot180 : VectorComplexMLAPred<(u32 2)>;
  defm _rot270 : VectorComplexMLAPred<(u32 3)>;
}

let params = T.Usual in
defm vcaddq : VectorComplexAddAngle<(u32 1)>;

let params = T.Signed in
defm vhcaddq : VectorComplexAddAngle<(u32 0)>;

let params = T.Float in {
defm vcmulq : VectorComplexMulAngle;
defm vcmlaq : VectorComplexMLAAngle;
}

multiclass MVEBinaryVectorHoriz32<dag subtract, dag exchange, string xsuffix> {
  def xsuffix#"q"
    : Intrinsic<Scalar32, (args Vector:$a, Vector:$b),
                          (IRInt<"vmldava", [Vector]>
                           (unsignedflag Scalar), subtract, exchange,
                           (zeroinit Scalar32), $a, $b)>;
  def xsuffix#"q_p"
    : Intrinsic<Scalar32, (args Vector:$a, Vector:$b, Predicate:$pred),
                          (IRInt<"vmldava_predicated", [Vector, Predicate]>
                           (unsignedflag Scalar), subtract, exchange,
                           (zeroinit Scalar32), $a, $b, $pred)>;

  def "a"#xsuffix#"q"
    : Intrinsic<Scalar32, (args Scalar32:$a, Vector:$b, Vector:$c),
                          (IRInt<"vmldava", [Vector]>
                           (unsignedflag Scalar), subtract, exchange,
                           $a, $b, $c)>;
  def "a"#xsuffix#"q_p"
    : Intrinsic<Scalar32, (args Scalar32:$a, Vector:$b, Vector:$c,
                                Predicate:$pred),
                          (IRInt<"vmldava_predicated", [Vector, Predicate]>
                           (unsignedflag Scalar), subtract, exchange,
                           $a, $b, $c, $pred)>;
}

class IntrSplit64<Type resty, dag args, dag codegen>
  : Intrinsic<resty, args,
              (seq (u32 (lshr $a, (u64 32))):$hi,
                   (u32 $a):$lo,
                   codegen:$pair,
                   (or (shl (u64 (xval $pair, 1)), (u64 32)),
                            (u64 (xval $pair, 0))))>;

class IntrSplit64ZeroInit<Type resty, dag args, dag codegen>
  : Intrinsic<resty, args,
              (seq (zeroinit u32):$hi,
                   (zeroinit u32):$lo,
                   codegen:$pair,
                   (or (shl (u64 (xval $pair, 1)), (u64 32)),
                            (u64 (xval $pair, 0))))>;

multiclass MVEBinaryVectorHoriz64Base<dag subtract, dag exchange,
                                      string xsuffix, string irname> {
  def xsuffix#"q"
    : IntrSplit64ZeroInit<Scalar64, (args Vector:$a, Vector:$b),
                          (IRInt<irname, [Vector]>
                           (unsignedflag Scalar), subtract, exchange,
                           $lo, $hi, $a, $b)>;
  def xsuffix#"q_p"
    : IntrSplit64ZeroInit<Scalar64, (args Vector:$a, Vector:$b,
                                          Predicate:$pred),
                          (IRInt<irname#"_predicated", [Vector, Predicate]>
                           (unsignedflag Scalar), subtract, exchange,
                           $lo, $hi, $a, $b, $pred)>;

  def "a"#xsuffix#"q"
    : IntrSplit64<Scalar64, (args Scalar64:$a, Vector:$b, Vector:$c),
                          (IRInt<irname, [Vector]>
                           (unsignedflag Scalar), subtract, exchange,
                           $lo, $hi, $b, $c)>;
  def "a"#xsuffix#"q_p"
    : IntrSplit64<Scalar64, (args Scalar64:$a, Vector:$b, Vector:$c,
                                  Predicate:$pred),
                          (IRInt<irname#"_predicated", [Vector, Predicate]>
                           (unsignedflag Scalar), subtract, exchange,
                           $lo, $hi, $b, $c, $pred)>;
}

multiclass MVEBinaryVectorHoriz64<dag subtract, dag exchange, string xsuffix> {
  defm "" : MVEBinaryVectorHoriz64Base<subtract, exchange, xsuffix, "vmlldava">;
}

multiclass MVEBinaryVectorHoriz64R<dag subtract, dag exchange, string xsuffix> {
  defm "" : MVEBinaryVectorHoriz64Base<subtract, exchange, xsuffix,
                                       "vrmlldavha">;
}

let params = T.Int in {
def vabavq : Intrinsic<u32, (args u32:$a, Vector:$b, Vector:$c),
    (IRInt<"vabav", [Vector]> (unsignedflag Scalar), $a, $b, $c)>;
def vabavq_p : Intrinsic<u32, (args u32:$a, Vector:$b, Vector:$c,
                                    Predicate:$pred),
    (IRInt<"vabav_predicated", [Vector, Predicate]>
                               (unsignedflag Scalar), $a, $b, $c, $pred)>;

defm vmladav  : MVEBinaryVectorHoriz32<V.False, V.False, "">;
}

let params = T.Signed in {
defm vmladav : MVEBinaryVectorHoriz32<V.False, V.True, "x">;
defm vmlsdav : MVEBinaryVectorHoriz32<V.True, V.False, "">;
defm vmlsdav : MVEBinaryVectorHoriz32<V.True, V.True, "x">;
}

let params = [u16, s16, u32, s32] in
defm vmlaldav : MVEBinaryVectorHoriz64<V.False, V.False, "">;

let params = [s16, s32] in {
defm vmlaldav : MVEBinaryVectorHoriz64<V.False, V.True, "x">;
defm vmlsldav : MVEBinaryVectorHoriz64<V.True, V.False, "">;
defm vmlsldav : MVEBinaryVectorHoriz64<V.True, V.True, "x">;
}

let params = T.Int32 in
defm vrmlaldavh : MVEBinaryVectorHoriz64R<V.False, V.False, "">;

let params = [s32] in {
defm vrmlaldavh : MVEBinaryVectorHoriz64R<V.False, V.True, "x">;
defm vrmlsldavh : MVEBinaryVectorHoriz64R<V.True, V.False, "">;
defm vrmlsldavh : MVEBinaryVectorHoriz64R<V.True, V.True, "x">;
}

foreach desttype = T.All in {
  // We want a vreinterpretq between every pair of supported vector types
  // _except_ that there shouldn't be one from a type to itself.
  //
  // So this foldl expression implements what you'd write in Python as
  // [srctype for srctype in T.All if srctype != desttype]
  let params = !foldl([]<Type>, T.All, tlist, srctype, !listconcat(tlist,
      !if(!eq(!cast<string>(desttype),!cast<string>(srctype)),[],[srctype])))
  in {
    def "vreinterpretq_" # desttype: Intrinsic<
        VecOf<desttype>, (args Vector:$x), (bitcast $x, VecOf<desttype>)>;
  }
}

let params = T.All in {
  let pnt = PNT_None in {
    def vcreateq: Intrinsic<Vector, (args u64:$a, u64:$b),
        (bitcast (ielt_const (ielt_const (undef VecOf<u64>), $a, 0),
                             $b, 1), Vector)>;
    def vuninitializedq: Intrinsic<Vector, (args), (undef Vector)>;
  }

  // This is the polymorphic form of vuninitializedq, which takes no type
  // suffix, but takes an _unevaluated_ vector parameter and returns an
  // uninitialized vector of the same vector type.
  //
  // This intrinsic has no _non_-polymorphic form exposed to the user. But each
  // separately typed version of it still has to have its own clang builtin id,
  // which can't be called vuninitializedq_u32 or similar because that would
  // collide with the explicit nullary versions above. So I'm calling them
  // vuninitializedq_polymorphic_u32 (and so on) for builtin id purposes; that
  // full name never appears in the header file due to the polymorphicOnly
  // flag, and the _polymorphic suffix is omitted from the shortened name by
  // the custom PolymorphicNameType here.
  let polymorphicOnly = 1, nonEvaluating = 1,
      pnt = PolymorphicNameType<1, "polymorphic"> in {
    def vuninitializedq_polymorphic: Intrinsic<
        Vector, (args Vector), (undef Vector)>;
  }

  def vgetq_lane: Intrinsic<Scalar, (args Vector:$v, imm_lane:$lane),
                            (xelt_var $v, $lane)>;
  def vsetq_lane: Intrinsic<Vector, (args unpromoted<Scalar>:$e, Vector:$v, imm_lane:$lane),
                            (ielt_var $v, $e, $lane)>;
}
