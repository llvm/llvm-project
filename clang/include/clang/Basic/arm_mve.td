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

let params = [Void], pnt = PNT_None in
def urshrl: Intrinsic<u64, (args u64:$value, imm_1to32:$shift),
                      (seq (u32 (lshr $value, (u64 32))):$hi,
                           (u32 $value):$lo,
                           (IRInt<"urshrl"> $lo, $hi, $shift):$pair,
                           (or (shl (u64 (xval $pair, 1)), (u64 32)),
                               (u64 (xval $pair, 0))))>;

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
