// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +xop -emit-cir -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +xop -fno-signed-char -emit-cir -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +xop -fclangir -emit-llvm -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +xop -fno-signed-char -fclangir -emit-llvm -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +xop -emit-cir -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +xop -fno-signed-char -emit-cir -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +xop -fclangir -emit-llvm -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +xop -fno-signed-char -fclangir -emit-llvm -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +xop -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=OGCG
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +xop -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +xop -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +xop -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=OGCG

#include <x86intrin.h>

// This test mimics clang/test/CodeGen/X86/xop-builtins.c, which eventually
// CIR shall be able to support fully.

__m128i test_mm_roti_epi8(__m128i a) {
  // CIR-LABEL: test_mm_roti_epi8
  // CIR: cir.vec.splat %{{.*}} : !{{[us]}}8i, !cir.vector<16 x !{{[us]}}8i> 
  // CIR: cir.call_llvm_intrinsic "fshl" %{{.*}} : (!cir.vector<16 x !{{[su]}}8i>, !cir.vector<16 x !{{[su]}}8i>, !cir.vector<16 x !{{[su]}}8i>) -> !cir.vector<16 x !{{[su]}}8i> 
  
  // LLVM-LABEL: test_mm_roti_epi8
  // LLVM: %[[CASTED_VAR:.*]] = bitcast <2 x i64> %{{.*}} to <16 x i8>
  // LLVM: call <16 x i8> @llvm.fshl.v16i8(<16 x i8> %[[CASTED_VAR]], <16 x i8> %[[CASTED_VAR]], <16 x i8> splat (i8 1))
  
  // OGCG-LABEL: test_mm_roti_epi8
  // OGCG: %[[CASTED_VAR:.*]] = bitcast <2 x i64> %{{.*}} to <16 x i8>
  // OGCG: call <16 x i8> @llvm.fshl.v16i8(<16 x i8> %[[CASTED_VAR]], <16 x i8> %[[CASTED_VAR]], <16 x i8> splat (i8 1))
  return _mm_roti_epi8(a, 1);
}

__m128i test_mm_roti_epi16(__m128i a) {
  // CIR-LABEL: test_mm_roti_epi16
  // CIR: cir.vec.splat %{{.*}} : !{{[us]}}16i, !cir.vector<8 x !u16i> 
  // CIR: cir.call_llvm_intrinsic "fshl" %{{.*}} : (!cir.vector<8 x !{{[su]}}16i>, !cir.vector<8 x !{{[su]}}16i>, !cir.vector<8 x !u16i>) -> !cir.vector<8 x !{{[su]}}16i> 
  
  // LLVM-LABEL: test_mm_roti_epi16
  // LLVM: %[[CASTED_VAR:.*]] = bitcast <2 x i64> %{{.*}} to <8 x i16>
  // LLVM: call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %[[CASTED_VAR]], <8 x i16> %[[CASTED_VAR]], <8 x i16> splat (i16 50))
  
  // OGCG-LABEL: test_mm_roti_epi16
  // OGCG: %[[CASTED_VAR:.*]] = bitcast <2 x i64> %{{.*}} to <8 x i16>
  // OGCG: call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %[[CASTED_VAR]], <8 x i16> %[[CASTED_VAR]], <8 x i16> splat (i16 50))
  return _mm_roti_epi16(a, 50);
 }

__m128i test_mm_roti_epi32(__m128i a) {
  // CIR-LABEL: test_mm_roti_epi32
  // CIR: cir.vec.splat %{{.*}} : !{{[us]}}32i, !cir.vector<4 x !u32i> 
  // CIR: cir.call_llvm_intrinsic "fshl" %{{.*}} : (!cir.vector<4 x !{{[su]}}32i>, !cir.vector<4 x !{{[su]}}32i>, !cir.vector<4 x !u32i>) -> !cir.vector<4 x !{{[su]}}32i> 
  
  // LLVM-LABEL: test_mm_roti_epi32
  // LLVM: %[[CASTED_VAR:.*]] = bitcast <2 x i64> %{{.*}} to <4 x i32>
  // LLVM: call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %[[CASTED_VAR]], <4 x i32> %[[CASTED_VAR]], <4 x i32> splat (i32 226))
  
  // OGCG-LABEL: test_mm_roti_epi32
  // OGCG: %[[CASTED_VAR:.*]] = bitcast <2 x i64> %{{.*}} to <4 x i32>
  // OGCG: call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %[[CASTED_VAR]], <4 x i32> %[[CASTED_VAR]], <4 x i32> splat (i32 226))
  return _mm_roti_epi32(a, -30);
 }

__m128i test_mm_roti_epi64(__m128i a) {
  // CIR-LABEL: test_mm_roti_epi64
  // CIR: cir.vec.splat %{{.*}} : !u64i, !cir.vector<2 x !u64i> 
  // CIR: cir.call_llvm_intrinsic "fshl" %{{.*}} : (!cir.vector<2 x !{{[su]}}64i>, !cir.vector<2 x !{{[su]}}64i>, !cir.vector<2 x !u64i>) -> !cir.vector<2 x !s64i> 
  
  // LLVM-LABEL: test_mm_roti_epi64
  // LLVM: %[[VAR:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
  // LLVM: call <2 x i64> @llvm.fshl.v2i64(<2 x i64> %[[VAR]], <2 x i64> %[[VAR]], <2 x i64> splat (i64 100))
  
  // OGCG-LABEL: test_mm_roti_epi64
  // OGCG: %[[VAR:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
  // OGCG: call <2 x i64> @llvm.fshl.v2i64(<2 x i64> %[[VAR]], <2 x i64> %[[VAR]], <2 x i64> splat (i64 100))
  return _mm_roti_epi64(a, 100);
 }

__m128i test_mm_com_epu8(__m128i a, __m128i b) {
  // CIR-LABEL: test_mm_com_epu8
  // CIR: %[[CMP:.*]] = cir.vec.cmp(lt, %{{.*}}, %{{.*}}) : !cir.vector<16 x !u8i>, !cir.vector<16 x !s8i>
  // CIR: %[[RES:.*]] = cir.cast bitcast %[[CMP]] : !cir.vector<16 x !s8i> -> !cir.vector<2 x !s64i>

  // LLVM-LABEL: test_mm_com_epu8
  // LLVM: %[[CMP:.*]] = icmp ult <16 x i8> %{{.*}}, %{{.*}}
  // LLVM: %[[RES:.*]] = sext <16 x i1> %[[CMP]] to <16 x i8>
  // LLVM: %{{.*}} = bitcast <16 x i8> %[[RES]] to <2 x i64>

  // OGCG-LABEL: test_mm_com_epu8
  // OGCG: %[[CMP:.*]] = icmp ult <16 x i8> %{{.*}}, %{{.*}}
  // OGCG: %[[RES:.*]] = sext <16 x i1> %[[CMP]] to <16 x i8>
  // OGCG: %{{.*}} = bitcast <16 x i8> %[[RES]] to <2 x i64>
  return _mm_com_epu8(a, b, 0);
}

__m128i test_mm_com_epu16(__m128i a, __m128i b) {
  // CIR-LABEL: test_mm_com_epu16
  // CIR: %[[VAL1:.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s16i> -> !cir.vector<8 x !u16i>
  // CIR: %[[VAL2:.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s16i> -> !cir.vector<8 x !u16i>
  // CIR: %[[CMP:.*]] = cir.vec.cmp(lt, %[[VAL1]], %[[VAL2]]) : !cir.vector<8 x !u16i>, !cir.vector<8 x !s16i>
  // CIR: %[[RES:.*]] = cir.cast bitcast %[[CMP]] : !cir.vector<8 x !s16i> -> !cir.vector<2 x !s64i>

  // LLVM-LABEL: test_mm_com_epu16
  // LLVM: %[[CMP:.*]] = icmp ult <8 x i16> %{{.*}}, %{{.*}}
  // LLVM: %[[RES:.*]] = sext <8 x i1> %[[CMP]] to <8 x i16>
  // LLVM: %{{.*}} = bitcast <8 x i16> %[[RES]] to <2 x i64>

  // OGCG-LABEL: test_mm_com_epu16
  // OGCG: %[[CMP:.*]] = icmp ult <8 x i16> %{{.*}}, %{{.*}}
  // OGCG: %[[RES:.*]] = sext <8 x i1> %[[CMP]] to <8 x i16>
  // OGCG: %{{.*}} = bitcast <8 x i16> %[[RES]] to <2 x i64>
  return _mm_com_epu16(a, b, 0);
}

__m128i test_mm_com_epu32(__m128i a, __m128i b) {
  // CIR-LABEL: test_mm_com_epu32
  // CIR: %[[VAL1:.*]] = cir.cast bitcast %{{.*}} : !cir.vector<4 x !s32i> -> !cir.vector<4 x !u32i>
  // CIR: %[[VAL2:.*]] = cir.cast bitcast %{{.*}} : !cir.vector<4 x !s32i> -> !cir.vector<4 x !u32i>
  // CIR: %[[CMP:.*]] = cir.vec.cmp(lt, %[[VAL1]], %[[VAL2]]) : !cir.vector<4 x !u32i>, !cir.vector<4 x !s32i>
  // CIR: %[[RES:.*]] = cir.cast bitcast %[[CMP]] : !cir.vector<4 x !s32i> -> !cir.vector<2 x !s64i>

  // LLVM-LABEL: test_mm_com_epu32
  // LLVM: %[[CMP:.*]] = icmp ult <4 x i32> %{{.*}}, %{{.*}}
  // LLVM: %[[RES:.*]] = sext <4 x i1> %[[CMP]] to <4 x i32>
  // LLVM: %{{.*}} = bitcast <4 x i32> %[[RES]] to <2 x i64>

  // OGCG-LABEL: test_mm_com_epu32
  // OGCG: %[[CMP:.*]] = icmp ult <4 x i32> %{{.*}}, %{{.*}}
  // OGCG: %[[RES:.*]] = sext <4 x i1> %[[CMP]] to <4 x i32>
  // OGCG: %{{.*}} = bitcast <4 x i32> %[[RES]] to <2 x i64>
  return _mm_com_epu32(a, b, 0);
}

__m128i test_mm_com_epu64(__m128i a, __m128i b) {
  // CIR-LABEL: test_mm_com_epu64
  // CIR: %[[VAL1:.*]] = cir.cast bitcast %{{.*}} : !cir.vector<2 x !s64i> -> !cir.vector<2 x !u64i>
  // CIR: %[[VAL2:.*]] = cir.cast bitcast %{{.*}} : !cir.vector<2 x !s64i> -> !cir.vector<2 x !u64i>
  // CIR: %[[CMP:.*]] = cir.vec.cmp(lt, %[[VAL1]], %[[VAL2]]) : !cir.vector<2 x !u64i>, !cir.vector<2 x !s64i>

  // LLVM-LABEL: test_mm_com_epu64
  // LLVM: %[[CMP:.*]] = icmp ult <2 x i64> %{{.*}}, %{{.*}}
  // LLVM: %[[RES:.*]] = sext <2 x i1> %[[CMP]] to <2 x i64>

  // OGCG-LABEL: test_mm_com_epu64
  // OGCG: %[[CMP:.*]] = icmp ult <2 x i64> %{{.*}}, %{{.*}}
  // OGCG: %[[RES:.*]] = sext <2 x i1> %[[CMP]] to <2 x i64>
  a = _mm_com_epu64(a, b, 0);

  // CIR: %[[VAL1:.*]] = cir.cast bitcast %{{.*}} : !cir.vector<2 x !s64i> -> !cir.vector<2 x !u64i>
  // CIR: %[[VAL2:.*]] = cir.cast bitcast %{{.*}} : !cir.vector<2 x !s64i> -> !cir.vector<2 x !u64i>
  // CIR: %[[CMP:.*]] = cir.vec.cmp(le, %[[VAL1]], %[[VAL2]]) : !cir.vector<2 x !u64i>, !cir.vector<2 x !s64i>

  // LLVM: %[[CMP:.*]] = icmp ule <2 x i64> %{{.*}}, %{{.*}}
  // LLVM: %[[RES:.*]] = sext <2 x i1> %[[CMP]] to <2 x i64>

  // OGCG: %[[CMP:.*]] = icmp ule <2 x i64> %{{.*}}, %{{.*}}
  // OGCG: %[[RES:.*]] = sext <2 x i1> %[[CMP]] to <2 x i64>
  a = _mm_com_epu64(a, b, 1);

  // CIR: %[[VAL1:.*]] = cir.cast bitcast %{{.*}} : !cir.vector<2 x !s64i> -> !cir.vector<2 x !u64i>
  // CIR: %[[VAL2:.*]] = cir.cast bitcast %{{.*}} : !cir.vector<2 x !s64i> -> !cir.vector<2 x !u64i>
  // CIR: %[[CMP:.*]] = cir.vec.cmp(gt, %[[VAL1]], %[[VAL2]]) : !cir.vector<2 x !u64i>, !cir.vector<2 x !s64i>

  // LLVM: %[[CMP:.*]] = icmp ugt <2 x i64> %{{.*}}, %{{.*}}
  // LLVM: %[[RES:.*]] = sext <2 x i1> %[[CMP]] to <2 x i64>

  // OGCG: %[[CMP:.*]] = icmp ugt <2 x i64> %{{.*}}, %{{.*}}
  // OGCG: %[[RES:.*]] = sext <2 x i1> %[[CMP]] to <2 x i64>
  a = _mm_com_epu64(a, b, 2);

  // CIR: %[[VAL1:.*]] = cir.cast bitcast %{{.*}} : !cir.vector<2 x !s64i> -> !cir.vector<2 x !u64i>
  // CIR: %[[VAL2:.*]] = cir.cast bitcast %{{.*}} : !cir.vector<2 x !s64i> -> !cir.vector<2 x !u64i>
  // CIR: %[[CMP:.*]] = cir.vec.cmp(ge, %[[VAL1]], %[[VAL2]]) : !cir.vector<2 x !u64i>, !cir.vector<2 x !s64i>

  // LLVM: %[[CMP:.*]] = icmp uge <2 x i64> %{{.*}}, %{{.*}}
  // LLVM: %[[RES:.*]] = sext <2 x i1> %[[CMP]] to <2 x i64>

  // OGCG: %[[CMP:.*]] = icmp uge <2 x i64> %{{.*}}, %{{.*}}
  // OGCG: %[[RES:.*]] = sext <2 x i1> %[[CMP]] to <2 x i64>
  a = _mm_com_epu64(a, b, 3);

  // CIR: %[[VAL1:.*]] = cir.cast bitcast %{{.*}} : !cir.vector<2 x !s64i> -> !cir.vector<2 x !u64i>
  // CIR: %[[VAL2:.*]] = cir.cast bitcast %{{.*}} : !cir.vector<2 x !s64i> -> !cir.vector<2 x !u64i>
  // CIR: %[[CMP:.*]] = cir.vec.cmp(eq, %[[VAL1]], %[[VAL2]]) : !cir.vector<2 x !u64i>, !cir.vector<2 x !s64i>

  // LLVM: %[[CMP:.*]] = icmp eq <2 x i64> %{{.*}}, %{{.*}}
  // LLVM: %[[RES:.*]] = sext <2 x i1> %[[CMP]] to <2 x i64>

  // OGCG: %[[CMP:.*]] = icmp eq <2 x i64> %{{.*}}, %{{.*}}
  // OGCG: %[[RES:.*]] = sext <2 x i1> %[[CMP]] to <2 x i64>
  a = _mm_com_epu64(a, b, 4);

  // CIR: %[[VAL1:.*]] = cir.cast bitcast %{{.*}} : !cir.vector<2 x !s64i> -> !cir.vector<2 x !u64i>
  // CIR: %[[VAL2:.*]] = cir.cast bitcast %{{.*}} : !cir.vector<2 x !s64i> -> !cir.vector<2 x !u64i>
  // CIR: %[[CMP:.*]] = cir.vec.cmp(ne, %[[VAL1]], %[[VAL2]]) : !cir.vector<2 x !u64i>, !cir.vector<2 x !s64i>

  // LLVM: %[[CMP:.*]] = icmp ne <2 x i64> %{{.*}}, %{{.*}}
  // LLVM: %[[RES:.*]] = sext <2 x i1> %[[CMP]] to <2 x i64>

  // OGCG: %[[CMP:.*]] = icmp ne <2 x i64> %{{.*}}, %{{.*}}
  // OGCG: %[[RES:.*]] = sext <2 x i1> %[[CMP]] to <2 x i64>
  return _mm_com_epu64(a, b, 5);
}

__m128i test_mm_com_epi8(__m128i a, __m128i b) {
  // CIR-LABEL: test_mm_com_epi8
  // CIR: %[[CMP:.*]] = cir.vec.cmp(lt, %{{.*}}, %{{.*}}) : !cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>
  // CIR: %[[RES:.*]] = cir.cast bitcast %[[CMP]] : !cir.vector<16 x !s8i> -> !cir.vector<2 x !s64i>

  // LLVM-LABEL: test_mm_com_epi8
  // LLVM: %[[CMP:.*]] = icmp slt <16 x i8> %{{.*}}, %{{.*}}
  // LLVM: %[[RES:.*]] = sext <16 x i1> %[[CMP]] to <16 x i8>
  // LLVM: %{{.*}} = bitcast <16 x i8> %[[RES]] to <2 x i64>

  // OGCG-LABEL: test_mm_com_epi8
  // OGCG: %[[CMP:.*]] = icmp slt <16 x i8> %{{.*}}, %{{.*}}
  // OGCG: %[[RES:.*]] = sext <16 x i1> %[[CMP]] to <16 x i8>
  // OGCG: %{{.*}} = bitcast <16 x i8> %[[RES]] to <2 x i64>
  return _mm_com_epi8(a, b, 0);
}

__m128i test_mm_com_epi16(__m128i a, __m128i b) {
  // CIR-LABEL: test_mm_com_epi16
  // CIR: %[[CMP:.*]] = cir.vec.cmp(lt, %{{.*}}, %{{.*}}) : !cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>
  // CIR: %[[RES:.*]] = cir.cast bitcast %[[CMP]] : !cir.vector<8 x !s16i> -> !cir.vector<2 x !s64i>

  // LLVM-LABEL: test_mm_com_epi16
  // LLVM: %[[CMP:.*]] = icmp slt <8 x i16> %{{.*}}, %{{.*}}
  // LLVM: %[[RES:.*]] = sext <8 x i1> %[[CMP]] to <8 x i16>
  // LLVM: %{{.*}} = bitcast <8 x i16> %[[RES]] to <2 x i64>

  // OGCG-LABEL: test_mm_com_epi16
  // OGCG: %[[CMP:.*]] = icmp slt <8 x i16> %{{.*}}, %{{.*}}
  // OGCG: %[[RES:.*]] = sext <8 x i1> %[[CMP]] to <8 x i16>
  // OGCG: %{{.*}} = bitcast <8 x i16> %[[RES]] to <2 x i64>
  return _mm_com_epi16(a, b, 0);
}

__m128i test_mm_com_epi32(__m128i a, __m128i b) {
  // CIR-LABEL: test_mm_com_epi32
  // CIR: %[[CMP:.*]] = cir.vec.cmp(lt, %{{.*}}, %{{.*}}) : !cir.vector<4 x !s32i>, !cir.vector<4 x !s32i>
  // CIR: %[[RES:.*]] = cir.cast bitcast %[[CMP]] : !cir.vector<4 x !s32i> -> !cir.vector<2 x !s64i>

  // LLVM-LABEL: test_mm_com_epi32
  // LLVM: %[[CMP:.*]] = icmp slt <4 x i32> %{{.*}}, %{{.*}}
  // LLVM: %[[RES:.*]] = sext <4 x i1> %[[CMP]] to <4 x i32>
  // LLVM: %{{.*}} = bitcast <4 x i32> %[[RES]] to <2 x i64>

  // OGCG-LABEL: test_mm_com_epi32
  // OGCG: %[[CMP:.*]] = icmp slt <4 x i32> %{{.*}}, %{{.*}}
  // OGCG: %[[RES:.*]] = sext <4 x i1> %[[CMP]] to <4 x i32>
  // OGCG: %{{.*}} = bitcast <4 x i32> %[[RES]] to <2 x i64>
  return _mm_com_epi32(a, b, 0);
}

__m128i test_mm_com_epi64(__m128i a, __m128i b) {
  // CIR-LABEL: test_mm_com_epi64
  // CIR: %[[CMP:.*]] = cir.vec.cmp(lt, %{{.*}}, %{{.*}}) : !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>

  // LLVM-LABEL: test_mm_com_epi64
  // LLVM: %[[CMP:.*]] = icmp slt <2 x i64> %{{.*}}, %{{.*}}
  // LLVM: %[[RES:.*]] = sext <2 x i1> %[[CMP]] to <2 x i64>

  // OGCG-LABEL: test_mm_com_epi64
  // OGCG: %[[CMP:.*]] = icmp slt <2 x i64> %{{.*}}, %{{.*}}
  // OGCG: %[[RES:.*]] = sext <2 x i1> %[[CMP]] to <2 x i64>
  a = _mm_com_epi64(a, b, 0);

  // CIR: %[[CMP1:.*]] = cir.vec.cmp(le, %{{.*}}, %{{.*}}) : !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>

  // LLVM: %[[CMP1:.*]] = icmp sle <2 x i64> %{{.*}}, %{{.*}}
  // LLVM: %[[RES1:.*]] = sext <2 x i1> %[[CMP1]] to <2 x i64>

  // OGCG: %[[CMP1:.*]] = icmp sle <2 x i64> %{{.*}}, %{{.*}}
  // OGCG: %[[RES1:.*]] = sext <2 x i1> %[[CMP1]] to <2 x i64>
  a = _mm_com_epi64(a, b, 1);

  // CIR: %[[CMP1:.*]] = cir.vec.cmp(gt, %{{.*}}, %{{.*}}) : !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>

  // LLVM: %[[CMP1:.*]] = icmp sgt <2 x i64> %{{.*}}, %{{.*}}
  // LLVM: %[[RES1:.*]] = sext <2 x i1> %[[CMP1]] to <2 x i64>

  // OGCG: %[[CMP1:.*]] = icmp sgt <2 x i64> %{{.*}}, %{{.*}}
  // OGCG: %[[RES1:.*]] = sext <2 x i1> %[[CMP1]] to <2 x i64>
  a = _mm_com_epi64(a, b, 2);

  // CIR: %[[CMP1:.*]] = cir.vec.cmp(ge, %{{.*}}, %{{.*}}) : !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>

  // LLVM: %[[CMP1:.*]] = icmp sge <2 x i64> %{{.*}}, %{{.*}}
  // LLVM: %[[RES1:.*]] = sext <2 x i1> %[[CMP1]] to <2 x i64>

  // OGCG: %[[CMP1:.*]] = icmp sge <2 x i64> %{{.*}}, %{{.*}}
  // OGCG: %[[RES1:.*]] = sext <2 x i1> %[[CMP1]] to <2 x i64>
  a = _mm_com_epi64(a, b, 3);

  // CIR: %[[CMP1:.*]] = cir.vec.cmp(eq, %{{.*}}, %{{.*}}) : !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>

  // LLVM: %[[CMP1:.*]] = icmp eq <2 x i64> %{{.*}}, %{{.*}}
  // LLVM: %[[RES1:.*]] = sext <2 x i1> %[[CMP1]] to <2 x i64>

  // OGCG: %[[CMP1:.*]] = icmp eq <2 x i64> %{{.*}}, %{{.*}}
  // OGCG: %[[RES1:.*]] = sext <2 x i1> %[[CMP1]] to <2 x i64>
  a = _mm_com_epi64(a, b, 4);

  // CIR: %[[CMP1:.*]] = cir.vec.cmp(ne, %{{.*}}, %{{.*}}) : !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>

  // LLVM: %[[CMP1:.*]] = icmp ne <2 x i64> %{{.*}}, %{{.*}}
  // LLVM: %[[RES1:.*]] = sext <2 x i1> %[[CMP1]] to <2 x i64>

  // OGCG: %[[CMP1:.*]] = icmp ne <2 x i64> %{{.*}}, %{{.*}}
  // OGCG: %[[RES1:.*]] = sext <2 x i1> %[[CMP1]] to <2 x i64>
  return _mm_com_epi64(a, b, 5);
}

__m128i test_mm_com_epi32_false(__m128i a, __m128i b) {
  // CIR-LABEL: test_mm_com_epi32_false
  // CIR: %[[ZERO:.*]] = cir.const #cir.zero : !cir.vector<4 x !s32i>
  // CIR: %{{.*}} = cir.cast bitcast %[[ZERO]] : !cir.vector<4 x !s32i> -> !cir.vector<2 x !s64i>

  // LLVM-LABEL: test_mm_com_epi32_false
  // LLVM: store <2 x i64> zeroinitializer, ptr %[[A:.*]], align 16
  // LLVM: %[[ZERO:.*]] = load <2 x i64>, ptr %[[A]], align 16
  // LLVM: ret <2 x i64> %[[ZERO]]

  // OGCG-LABEL: test_mm_com_epi32_false
  // OGCG: ret <2 x i64> zeroinitializer
  return _mm_com_epi32(a, b, 6);
}

__m128i test_mm_com_epu32_false(__m128i a, __m128i b) {
  // CIR-LABEL: test_mm_com_epu32_false
  // CIR: %[[ZERO:.*]] = cir.const #cir.zero : !cir.vector<4 x !s32i>
  // CIR: %{{.*}} = cir.cast bitcast %[[ZERO]] : !cir.vector<4 x !s32i> -> !cir.vector<2 x !s64i>

  // LLVM-LABEL: test_mm_com_epu32_false
  // LLVM: store <2 x i64> zeroinitializer, ptr %[[A:.*]], align 16
  // LLVM: %[[ZERO:.*]] = load <2 x i64>, ptr %[[A]], align 16
  // LLVM: ret <2 x i64> %[[ZERO]]

  // OGCG-LABEL: test_mm_com_epu32_false
  // OGCG: ret <2 x i64> zeroinitializer
  return _mm_com_epu32(a, b, 6);
}

__m128i test_mm_com_epi32_true(__m128i a, __m128i b) {
  // CIR-LABEL: test_mm_com_epi32_true
  // CIR: %[[VAL:.*]] = cir.const #cir.int<-1> : !s32i
  // CIR: %[[SPLAT:.*]] = cir.vec.splat %[[VAL]] : !s32i, !cir.vector<4 x !s32i>
  // CIR: %{{.*}} = cir.cast bitcast %[[SPLAT]] : !cir.vector<4 x !s32i> -> !cir.vector<2 x !s64i>

  // LLVM-LABEL: test_mm_com_epi32_true
  // LLVM: store <2 x i64> splat (i64 -1), ptr %[[VAL:.*]], align 16
  // LLVM: %[[SPLAT:.*]] = load <2 x i64>, ptr %[[VAL]], align 16
  // LLVM: ret <2 x i64> %[[SPLAT]]

  // OGCG-LABEL: test_mm_com_epi32_true
  // OGCG: ret <2 x i64> splat (i64 -1)
  return _mm_com_epi32(a, b, 7);
}

__m128i test_mm_com_epu32_true(__m128i a, __m128i b) {
  // CIR-LABEL: test_mm_com_epu32_true
  // CIR: %[[VAL:.*]] = cir.const #cir.int<-1> : !s32i
  // CIR: %[[SPLAT:.*]] = cir.vec.splat %[[VAL]] : !s32i, !cir.vector<4 x !s32i>
  // CIR: %{{.*}} = cir.cast bitcast %[[SPLAT]] : !cir.vector<4 x !s32i> -> !cir.vector<2 x !s64i>

  // LLVM-LABEL: test_mm_com_epu32_true
  // LLVM: store <2 x i64> splat (i64 -1), ptr %[[VAL:.*]], align 16
  // LLVM: %[[SPLAT:.*]] = load <2 x i64>, ptr %[[VAL]], align 16
  // LLVM: ret <2 x i64> %[[SPLAT]]

  // OGCG-LABEL: test_mm_com_epu32_true
  // OGCG: ret <2 x i64> splat (i64 -1)
  return _mm_com_epu32(a, b, 7);
}
