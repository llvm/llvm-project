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
  // CIR: cir.cast integral %{{.*}} : !u8i -> !u16i
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
  // CIR: cir.cast integral %{{.*}} : !u8i -> !u32i
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
  // CIR: cir.cast integral %{{.*}} : !u8i -> !u64i
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
