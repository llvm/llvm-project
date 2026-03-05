// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse2 -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefixes=CIR-CHECK,CIR-X64 --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse2 -fno-signed-char -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefixes=CIR-CHECK,CIR-X64 --input-file=%t.cir %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse2 -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM-CHECK,LLVM-X64 --input-file=%t.ll %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse2 -fno-signed-char -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM-CHECK,LLVM-X64 --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse2 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse2 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=OGCG

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse2 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse2 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=OGCG

// This test mimics clang/test/CodeGen/X86/sse2-builtins.c, which eventually
// CIR shall be able to support fully.

#include <immintrin.h>

void test_mm_clflush(void* A) {
  // CIR-LABEL: test_mm_clflush
  // LLVM-LABEL: test_mm_clflush
  _mm_clflush(A);
  // CIR-CHECK: {{%.*}} = cir.llvm.intrinsic "x86.sse2.clflush" {{%.*}} : (!cir.ptr<!void>) -> !void
  // LLVM-CHECK: call void @llvm.x86.sse2.clflush(ptr {{%.*}})
}

__m128d test_mm_undefined_pd(void) {
  // CIR-X64-LABEL: _mm_undefined_pd
  // CIR-X64: %{{.*}} = cir.const #cir.zero : !cir.vector<!cir.double x 2>
  // CIR-X64: cir.return %{{.*}} : !cir.vector<!cir.double x 2>

  // LLVM-X64-LABEL: test_mm_undefined_pd
  // LLVM-X64: store <2 x double> zeroinitializer, ptr %[[A:.*]], align 16
  // LLVM-X64: %{{.*}} = load <2 x double>, ptr %[[A]], align 16
  // LLVM-X64: ret <2 x double> %{{.*}}
  return _mm_undefined_pd();
}

__m128i test_mm_undefined_si128(void) {
  // CIR-LABEL: _mm_undefined_si128
  // CIR-CHECK: %[[A:.*]] = cir.const #cir.zero : !cir.vector<!cir.double x 2>
  // CIR-CHECK: %{{.*}} = cir.cast bitcast %[[A]] : !cir.vector<!cir.double x 2> -> !cir.vector<!s64i x 2>
  // CIR-CHECK: cir.return %{{.*}} : !cir.vector<!s64i x 2>

  // LLVM-CHECK-LABEL: test_mm_undefined_si128
  // LLVM-CHECK: store <2 x i64> zeroinitializer, ptr %[[A:.*]], align 16
  // LLVM-CHECK: %{{.*}} = load <2 x i64>, ptr %[[A]], align 16
  // LLVM-CHECK: ret <2 x i64> %{{.*}}
  return _mm_undefined_si128();
}

// Lowering to pextrw requires optimization.
int test_mm_extract_epi16(__m128i A) {
    
  // CIR-CHECK-LABEL: test_mm_extract_epi16
  // CIR-CHECK %{{.*}} = cir.vec.extract %{{.*}}[%{{.*}} : {{!u32i|!u64i}}] : !cir.vector<!s16i x 8>
  // CIR-CHECK %{{.*}} = cir.cast integral %{{.*}} : !u16i -> !s32i

  // LLVM-CHECK-LABEL: test_mm_extract_epi16
  // LLVM-CHECK: extractelement <8 x i16> %{{.*}}, {{i32|i64}} 1
  // LLVM-CHECK: zext i16 %{{.*}} to i32
  return _mm_extract_epi16(A, 1);
}

void test_mm_lfence(void) {
  // CIR-CHECK-LABEL: test_mm_lfence
  // LLVM-CHECK-LABEL: test_mm_lfence
  _mm_lfence();
  // CIR-CHECK: {{%.*}} = cir.llvm.intrinsic "x86.sse2.lfence" : () -> !void
  // LLVM-CHECK: call void @llvm.x86.sse2.lfence()
}

void test_mm_mfence(void) {
  // CIR-CHECK-LABEL: test_mm_mfence
  // LLVM-CHECK-LABEL: test_mm_mfence
  _mm_mfence();
  // CIR-CHECK: {{%.*}} = cir.llvm.intrinsic "x86.sse2.mfence" : () -> !void
  // LLVM-CHECK: call void @llvm.x86.sse2.mfence()
}

__m128i test_mm_shufflelo_epi16(__m128i A) {
  // CIR-LABEL: _mm_shufflelo_epi16
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s16i x 8>) [#cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i] : !cir.vector<!s16i x 8>

  // LLVM-LABEL: test_mm_shufflelo_epi16
  // LLVM: shufflevector <8 x i16> %{{.*}}, <8 x i16> poison, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 4, i32 5, i32 6, i32 7>

  // OGCG-LABEL: test_mm_shufflelo_epi16
  // OGCG: shufflevector <8 x i16> %{{.*}}, <8 x i16> poison, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 4, i32 5, i32 6, i32 7>
  return _mm_shufflelo_epi16(A, 0);
}

__m128i test_mm_shufflehi_epi16(__m128i A) {
  // CIR-LABEL: _mm_shufflehi_epi16
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s16i x 8>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<4> : !s32i, #cir.int<4> : !s32i, #cir.int<4> : !s32i, #cir.int<4> : !s32i] : !cir.vector<!s16i x 8>

  // LLVM-LABEL: test_mm_shufflehi_epi16
  // LLVM: shufflevector <8 x i16> %{{.*}}, <8 x i16> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 4, i32 4, i32 4>

  // OGCG-LABEL: test_mm_shufflehi_epi16
  // OGCG: shufflevector <8 x i16> %{{.*}}, <8 x i16> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 4, i32 4, i32 4>
  return _mm_shufflehi_epi16(A, 0);
}

__m128d test_mm_shuffle_pd(__m128d A, __m128d B) {
  // CIR-LABEL: test_mm_shuffle_pd
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!cir.double x 2>) [#cir.int<1> : !s32i, #cir.int<2> : !s32i] : !cir.vector<!cir.double x 2>

  // CHECK-LABEL: test_mm_shuffle_pd
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 1, i32 2>

  // OGCG-LABEL: test_mm_shuffle_pd
  // OGCG: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 1, i32 2>
  return _mm_shuffle_pd(A, B, 1);
}

