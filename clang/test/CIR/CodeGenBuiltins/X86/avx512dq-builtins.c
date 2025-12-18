// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512dq -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512dq -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512dq -fno-signed-char -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512dq -fno-signed-char -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512dq -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512dq -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=OGCG

#include <immintrin.h>

__m512i test_mm512_movm_epi64(__mmask8 __A) {
  // CIR-LABEL: _mm512_movm_epi64
  // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
  // CIR: %{{.*}} = cir.cast integral %{{.*}} : !cir.vector<8 x !cir.int<u, 1>> -> !cir.vector<8 x !s64i>

  // LLVM-LABEL: @test_mm512_movm_epi64
  // LLVM: %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // LLVM: %{{.*}} = sext <8 x i1> %{{.*}} to <8 x i64>

  // OGCG-LABEL: {{.*}}test_mm512_movm_epi64{{.*}}(
  // OGCG: %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // OGCG: %{{.*}} = sext <8 x i1> %{{.*}} to <8 x i64>
  return _mm512_movm_epi64(__A); 
}


__mmask8 test_kadd_mask8(__mmask8 A, __mmask8 B) {
 // CIR-LABEL: _kadd_mask8
 // CIR: cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.call_llvm_intrinsic "x86.avx512.kadd.b"
 // CIR: cir.cast bitcast {{.*}} : !cir.vector<8 x !cir.int<u, 1>> -> !u8i

 // LLVM-LABEL: _kadd_mask8
 // LLVM: [[L:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
 // LLVM: [[R:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
 // LLVM: [[RES:%.*]] = call <8 x i1> @llvm.x86.avx512.kadd.b(<8 x i1> [[L]], <8 x i1> [[R]])
 // LLVM: bitcast <8 x i1> [[RES]] to i8

 // OGCG-LABEL: _kadd_mask8
 // OGCG: bitcast i8 %{{.*}} to <8 x i1>
 // OGCG: bitcast i8 %{{.*}} to <8 x i1>
 // OGCG: call <8 x i1> @llvm.x86.avx512.kadd.b
 // OGCG: bitcast <8 x i1> {{.*}} to i8
 return _kadd_mask8(A, B);
}

__mmask16 test_kadd_mask16(__mmask16 A, __mmask16 B) {
 // CIR-LABEL: _kadd_mask16
 // CIR: cir.cast bitcast {{.*}} : !u16i -> !cir.vector<16 x !cir.int<u, 1>>
 // CIR: cir.cast bitcast {{.*}} : !u16i -> !cir.vector<16 x !cir.int<u, 1>>
 // CIR: cir.call_llvm_intrinsic "x86.avx512.kadd.w"
 // CIR: cir.cast bitcast {{.*}} : !cir.vector<16 x !cir.int<u, 1>> -> !u16i

 // LLVM-LABEL: _kadd_mask16
 // LLVM: [[L:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
 // LLVM: [[R:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
 // LLVM: [[RES:%.*]] = call <16 x i1> @llvm.x86.avx512.kadd.w(<16 x i1> [[L]], <16 x i1> [[R]])
 // LLVM: bitcast <16 x i1> [[RES]] to i16

 // OGCG-LABEL: _kadd_mask16
 // OGCG: bitcast i16 %{{.*}} to <16 x i1>
 // OGCG: bitcast i16 %{{.*}} to <16 x i1>
 // OGCG: call <16 x i1> @llvm.x86.avx512.kadd.w
 // OGCG: bitcast <16 x i1> {{.*}} to i16
 return _kadd_mask16(A, B);
}

__mmask8 test_kand_mask8(__mmask8 A, __mmask8 B) {
 // CIR-LABEL: _kand_mask8
 // CIR: cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.binop(and, {{.*}}, {{.*}}) : !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.cast bitcast {{.*}} : !cir.vector<8 x !cir.int<u, 1>> -> !u8i

 // LLVM-LABEL: _kand_mask8
 // LLVM: [[L:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
 // LLVM: [[R:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
 // LLVM: [[RES:%.*]] = and <8 x i1> [[L]], [[R]]
 // LLVM: bitcast <8 x i1> [[RES]] to i8

 // OGCG-LABEL: _kand_mask8
 // OGCG: bitcast i8 %{{.*}} to <8 x i1>
 // OGCG: bitcast i8 %{{.*}} to <8 x i1>
 // OGCG: and <8 x i1>
 // OGCG: bitcast <8 x i1> {{.*}} to i8
 return _kand_mask8(A, B);
}


__mmask8 test_kandn_mask8(__mmask8 A, __mmask8 B) {
 // CIR-LABEL: _kandn_mask8
 // CIR: cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.unary(not, {{.*}}) : !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.binop(and, {{.*}}, {{.*}}) : !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.cast bitcast {{.*}} : !cir.vector<8 x !cir.int<u, 1>> -> !u8i

 // LLVM-LABEL: _kandn_mask8
 // LLVM: [[L:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
 // LLVM: [[R:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
 // LLVM: xor <8 x i1> [[L]], splat (i1 true)
 // LLVM: and <8 x i1>
 // LLVM: bitcast <8 x i1> {{.*}} to i8

 // OGCG-LABEL: _kandn_mask8
 // OGCG: bitcast i8 %{{.*}} to <8 x i1>
 // OGCG: bitcast i8 %{{.*}} to <8 x i1>
 // OGCG: xor <8 x i1>
 // OGCG: and <8 x i1>
 // OGCG: bitcast <8 x i1> {{.*}} to i8

 return _kandn_mask8(A, B);
}

__mmask8 test_kor_mask8(__mmask8 A, __mmask8 B) {
 // CIR-LABEL: _kor_mask8
 // CIR: cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.binop(or, {{.*}}, {{.*}}) : !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.cast bitcast {{.*}} : !cir.vector<8 x !cir.int<u, 1>> -> !u8i

 // LLVM-LABEL: _kor_mask8
 // LLVM: [[L:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
 // LLVM: [[R:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
 // LLVM: or <8 x i1> [[L]], [[R]]
 // LLVM: bitcast <8 x i1> {{.*}} to i8

 // OGCG-LABEL: _kor_mask8
 // OGCG: bitcast i8 %{{.*}} to <8 x i1>
 // OGCG: bitcast i8 %{{.*}} to <8 x i1>
 // OGCG: or <8 x i1>
 // OGCG: bitcast <8 x i1> {{.*}} to i8
 return _kor_mask8(A, B);
}

__mmask8 test_kxor_mask8(__mmask8 A, __mmask8 B) {
 // CIR-LABEL: _kxor_mask8
 // CIR: cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.binop(xor, {{.*}}, {{.*}}) : !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.cast bitcast {{.*}} : !cir.vector<8 x !cir.int<u, 1>> -> !u8i

 // LLVM-LABEL: _kxor_mask8
 // LLVM: [[L:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
 // LLVM: [[R:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
 // LLVM: xor <8 x i1> [[L]], [[R]]
 // LLVM: bitcast <8 x i1> {{.*}} to i8

 // OGCG-LABEL: _kxor_mask8
 // OGCG: bitcast i8 %{{.*}} to <8 x i1>
 // OGCG: bitcast i8 %{{.*}} to <8 x i1>
 // OGCG: xor <8 x i1>
 // OGCG: bitcast <8 x i1> {{.*}} to i8
 return _kxor_mask8(A, B);
}

__mmask8 test_kxnor_mask8(__mmask8 A, __mmask8 B) {
 // CIR-LABEL: _kxnor_mask8
 // CIR: cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.unary(not, {{.*}}) : !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.binop(xor, {{.*}}, {{.*}}) : !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.cast bitcast {{.*}} : !cir.vector<8 x !cir.int<u, 1>> -> !u8i

 // LLVM-LABEL: _kxnor_mask8
 // LLVM: [[L:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
 // LLVM: [[R:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
 // LLVM: [[NOT:%.*]] = xor <8 x i1> [[L]], splat (i1 true)
 // LLVM: [[RES:%.*]] = xor <8 x i1> [[NOT]], [[R]]
 // LLVM: bitcast <8 x i1> [[RES]] to i8

 // OGCG-LABEL: _kxnor_mask8
 // OGCG: bitcast i8 %{{.*}} to <8 x i1>
 // OGCG: bitcast i8 %{{.*}} to <8 x i1>
 // OGCG: xor <8 x i1>
 // OGCG: xor <8 x i1>
 // OGCG: bitcast <8 x i1> {{.*}} to i8
 return _kxnor_mask8(A, B);
}


__mmask8 test_knot_mask8(__mmask8 A) {
 // CIR-LABEL: _knot_mask8
 // CIR: cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.unary(not, {{.*}}) : !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.cast bitcast {{.*}} : !cir.vector<8 x !cir.int<u, 1>> -> !u8i

 // LLVM-LABEL: _knot_mask8
 // LLVM: [[L:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
 // LLVM: xor <8 x i1> [[L]], {{.*}}
 // LLVM: bitcast <8 x i1> {{.*}} to i8

 // OGCG-LABEL: _knot_mask8
 // OGCG: bitcast i8 %{{.*}} to <8 x i1>
 // OGCG: xor <8 x i1>
 // OGCG: bitcast <8 x i1> {{.*}} to i8
 return _knot_mask8(A);
}

// Multiple user-level mask helpers inline to this same kmov builtin.
// CIR does not implement any special lowering for those helpers.
//
// Therefore, testing the builtin (__builtin_ia32_kmov*) directly is
// sufficient to cover the CIR lowering behavior. Testing each helper
// individually would add no new CIR paths.

__mmask8 test_kmov_b(__mmask8 A) {
 // CIR-LABEL: test_kmov_b
 // CIR: cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.cast bitcast {{.*}} : !cir.vector<8 x !cir.int<u, 1>> -> !u8i

 // LLVM-LABEL: test_kmov_b
 // LLVM: bitcast i8 %{{.*}} to <8 x i1>
 // LLVM: bitcast <8 x i1> {{.*}} to i8

 // OGCG-LABEL: test_kmov_b
 // OGCG: bitcast i8 %{{.*}} to <8 x i1>
 // OGCG: bitcast <8 x i1> {{.*}} to i8
 return __builtin_ia32_kmovb(A);
}

unsigned char test_kortestc_mask8_u8(__mmask8 __A, __mmask8 __B) {
  // CIR-LABEL: _kortestc_mask8_u8
  // CIR: %[[ALL_ONES:.*]] = cir.const #cir.int<255> : !u8i
  // CIR: %[[LHS:.*]] = cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
  // CIR: %[[RHS:.*]] = cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
  // CIR: %[[OR:.*]] = cir.binop(or, %[[LHS]], %[[RHS]]) : !cir.vector<8 x !cir.int<u, 1>>
  // CIR: %[[OR_INT:.*]] = cir.cast bitcast %[[OR]] : !cir.vector<8 x !cir.int<u, 1>> -> !u8i
  // CIR: %[[CMP:.*]] = cir.cmp(eq, %[[OR_INT]], %[[ALL_ONES]]) : !u8i, !cir.bool
  // CIR: cir.cast bool_to_int %[[CMP]] : !cir.bool -> !s32i
  // CIR: cir.cast integral {{.*}} : !s32i -> !u8i

  // LLVM-LABEL: _kortestc_mask8_u8
  // LLVM: %[[LHS:.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // LLVM: %[[RHS:.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // LLVM: %[[OR:.*]] = or <8 x i1> %[[LHS]], %[[RHS]]
  // LLVM: %[[CAST:.*]] = bitcast <8 x i1> %[[OR]] to i8
  // LLVM: %[[CMP:.*]] = icmp eq i8 %[[CAST]], -1
  // LLVM: %[[ZEXT:.*]] = zext i1 %[[CMP]] to i32
  // LLVM: trunc i32 %[[ZEXT]] to i8

  // OGCG-LABEL: _kortestc_mask8_u8
  // OGCG: %[[LHS:.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // OGCG: %[[RHS:.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // OGCG: %[[OR:.*]] = or <8 x i1> %[[LHS]], %[[RHS]]
  // OGCG: %[[CAST:.*]] = bitcast <8 x i1> %[[OR]] to i8
  // OGCG: %[[CMP:.*]] = icmp eq i8 %[[CAST]], -1
  // OGCG: %[[ZEXT:.*]] = zext i1 %[[CMP]] to i32
  // OGCG: trunc i32 %[[ZEXT]] to i8
  return _kortestc_mask8_u8(__A,__B);
}

unsigned char test_ktestc_mask8_u8(__mmask8 A, __mmask8 B) {
  // CIR-LABEL: _ktestc_mask8_u8
  // CIR: %[[LHS:.*]] = cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
  // CIR: %[[RHS:.*]] = cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
  // CIR: %[[RES:.*]] = cir.call_llvm_intrinsic "x86.avx512.ktestc.b"
  // CIR: cir.cast integral %[[RES]] : {{.*}} -> !u8i

  // LLVM-LABEL: _ktestc_mask8_u8
  // LLVM: %[[LHS:.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // LLVM: %[[RHS:.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // LLVM: %[[RES:.*]] = call i32 @llvm.x86.avx512.ktestc.b(<8 x i1> %[[LHS]], <8 x i1> %[[RHS]])
  // LLVM: trunc i32 %[[RES]] to i8

  // OGCG-LABEL: _ktestc_mask8_u8
  // OGCG: %[[LHS:.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // OGCG: %[[RHS:.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // OGCG: %[[RES:.*]] = call i32 @llvm.x86.avx512.ktestc.b
  // OGCG: trunc i32 %[[RES]] to i8
  return _ktestc_mask8_u8(A, B);
}

unsigned char test_ktestz_mask8_u8(__mmask8 A, __mmask8 B) {
  // CIR-LABEL: _ktestz_mask8_u8
  // CIR: %[[LHS:.*]] = cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
  // CIR: %[[RHS:.*]] = cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
  // CIR: %[[RES:.*]] = cir.call_llvm_intrinsic "x86.avx512.ktestz.b"
  // CIR: cir.cast integral %[[RES]] : {{.*}} -> !u8i

  // LLVM-LABEL: _ktestz_mask8_u8
  // LLVM: %[[LHS:.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // LLVM: %[[RHS:.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // LLVM: %[[RES:.*]] = call i32 @llvm.x86.avx512.ktestz.b(<8 x i1> %[[LHS]], <8 x i1> %[[RHS]])
  // LLVM: trunc i32 %[[RES]] to i8

  // OGCG-LABEL: _ktestz_mask8_u8
  // OGCG: %[[LHS:.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // OGCG: %[[RHS:.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // OGCG: %[[RES:.*]] = call i32 @llvm.x86.avx512.ktestz.b
  // OGCG: trunc i32 %[[RES]] to i8
  return _ktestz_mask8_u8(A, B);
}

unsigned char test_ktestc_mask16_u8(__mmask16 A, __mmask16 B) {
  // CIR-LABEL: _ktestc_mask16_u8
  // CIR: %[[LHS:.*]] = cir.cast bitcast {{.*}} : !u16i -> !cir.vector<16 x !cir.int<u, 1>>
  // CIR: %[[RHS:.*]] = cir.cast bitcast {{.*}} : !u16i -> !cir.vector<16 x !cir.int<u, 1>>
  // CIR: %[[RES:.*]] = cir.call_llvm_intrinsic "x86.avx512.ktestc.w"
  // CIR: cir.cast integral %[[RES]] : {{.*}} -> !u8i

  // LLVM-LABEL: _ktestc_mask16_u8
  // LLVM: %[[LHS:.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: %[[RHS:.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: %[[RES:.*]] = call i32 @llvm.x86.avx512.ktestc.w(<16 x i1> %[[LHS]], <16 x i1> %[[RHS]])
  // LLVM: trunc i32 %[[RES]] to i8

  // OGCG-LABEL: _ktestc_mask16_u8
  // OGCG: %[[LHS:.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // OGCG: %[[RHS:.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // OGCG: %[[RES:.*]] = call i32 @llvm.x86.avx512.ktestc.w
  // OGCG: trunc i32 %[[RES]] to i8
  return _ktestc_mask16_u8(A, B);
}

unsigned char test_ktestz_mask16_u8(__mmask16 A, __mmask16 B) {
  // CIR-LABEL: _ktestz_mask16_u8
  // CIR: %[[LHS:.*]] = cir.cast bitcast {{.*}} : !u16i -> !cir.vector<16 x !cir.int<u, 1>>
  // CIR: %[[RHS:.*]] = cir.cast bitcast {{.*}} : !u16i -> !cir.vector<16 x !cir.int<u, 1>>
  // CIR: %[[RES:.*]] = cir.call_llvm_intrinsic "x86.avx512.ktestz.w"
  // CIR: cir.cast integral %[[RES]] : {{.*}} -> !u8i

  // LLVM-LABEL: _ktestz_mask16_u8
  // LLVM: %[[LHS:.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: %[[RHS:.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: %[[RES:.*]] = call i32 @llvm.x86.avx512.ktestz.w(<16 x i1> %[[LHS]], <16 x i1> %[[RHS]])
  // LLVM: trunc i32 %[[RES]] to i8

  // OGCG-LABEL: _ktestz_mask16_u8
  // OGCG: %[[LHS:.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // OGCG: %[[RHS:.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // OGCG: %[[RES:.*]] = call i32 @llvm.x86.avx512.ktestz.w
  // OGCG: trunc i32 %[[RES]] to i8
  return _ktestz_mask16_u8(A, B);
}

__mmask16 test_mm512_movepi32_mask(__m512i __A) {
  // CIR-LABEL: _mm512_movepi32_mask
  // CIR: cir.cast bitcast %{{.*}} : !cir.vector<8 x !s64i> -> !cir.vector<16 x !s32i>
  // CIR: [[CMP:%.*]] = cir.vec.cmp(lt, %{{.*}}, %{{.*}}) : !cir.vector<16 x !s32i>, !cir.vector<16 x !cir.int<s, 1>>
  // CIR: %{{.*}} = cir.cast bitcast [[CMP]] : !cir.vector<16 x !cir.int<s, 1>> -> !u16i

  // LLVM-LABEL: @test_mm512_movepi32_mask
  // LLVM: [[CMP:%.*]] = icmp slt <16 x i32> %{{.*}}, zeroinitializer
  // LLVM: bitcast <16 x i1> [[CMP]] to i16

  // OGCG-LABEL: {{.*}}test_mm512_movepi32_mask{{.*}}(
  // OGCG: [[CMP:%.*]] = icmp slt <16 x i32> %{{.*}}, zeroinitializer
  // OGCG: bitcast <16 x i1> [[CMP]] to i16
  return _mm512_movepi32_mask(__A); 
}

__mmask8 test_mm512_movepi64_mask(__m512i __A) {
  // CIR-LABEL: _mm512_movepi64_mask
  // CIR: [[CMP:%.*]] = cir.vec.cmp(lt, %{{.*}}, %{{.*}}) : !cir.vector<8 x !s64i>, !cir.vector<8 x !cir.int<s, 1>>
  // CIR: %{{.*}} = cir.cast bitcast [[CMP]] : !cir.vector<8 x !cir.int<s, 1>> -> !u8i

  // LLVM-LABEL: @test_mm512_movepi64_mask
  // LLVM: [[CMP:%.*]] = icmp slt <8 x i64> %{{.*}}, zeroinitializer
  // LLVM: bitcast <8 x i1> [[CMP]] to i8

  // OGCG-LABEL: {{.*}}test_mm512_movepi64_mask{{.*}}(
  // OGCG: [[CMP:%.*]] = icmp slt <8 x i64> %{{.*}}, zeroinitializer
  // OGCG: bitcast <8 x i1> [[CMP]] to i8
  return _mm512_movepi64_mask(__A);
}