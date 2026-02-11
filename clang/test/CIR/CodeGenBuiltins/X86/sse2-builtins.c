// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse2 -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefixes=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse2 -fno-signed-char -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefixes=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse2 -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse2 -fno-signed-char -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse2 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse2 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=OGCG

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse2 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse2 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=OGCG

// This test mimics clang/test/CodeGen/X86/sse2-builtins.c, which eventually
// CIR shall be able to support fully.

#include <immintrin.h>

__m128d test_mm_undefined_pd(void) {
  // CIR-LABEL: _mm_undefined_pd
  // CIR: %{{.*}} = cir.const #cir.zero : !cir.vector<2 x !cir.double>
  // CIR: cir.return %{{.*}} : !cir.vector<2 x !cir.double>

  // CIR-LABEL: cir.func {{.*}}test_mm_undefined_pd
  // CIR: call @_mm_undefined_pd

  // LLVM-LABEL: test_mm_undefined_pd
  // LLVM: store <2 x double> zeroinitializer, ptr %[[A:.*]], align 16
  // LLVM: %{{.*}} = load <2 x double>, ptr %[[A]], align 16
  // LLVM: ret <2 x double> %{{.*}}

  // OGCG-LABEL: test_mm_undefined_pd
  // OGCG: ret <2 x double> zeroinitializer
  return _mm_undefined_pd();
}

__m128i test_mm_undefined_si128(void) {
  // CIR-LABEL: _mm_undefined_si128
  // CIR: %[[A:.*]] = cir.const #cir.zero : !cir.vector<2 x !cir.double>
  // CIR: %{{.*}} = cir.cast bitcast %[[A]] : !cir.vector<2 x !cir.double> ->
  // CIR: cir.return %{{.*}} :

  // CIR-LABEL: cir.func {{.*}}test_mm_undefined_si128
  // CIR: call @_mm_undefined_si128

  // LLVM-LABEL: test_mm_undefined_si128
  // LLVM: store <2 x i64> zeroinitializer, ptr %[[A:.*]], align 16
  // LLVM: %{{.*}} = load <2 x i64>, ptr %[[A]], align 16
  // LLVM: ret <2 x i64> %{{.*}}

  // OGCG-LABEL: test_mm_undefined_si128
  // OGCG: ret <2 x i64> zeroinitializer
  return _mm_undefined_si128();
}

// Lowering to pextrw requires optimization.
int test_mm_extract_epi16(__m128i A) {
  // CIR-LABEL: test_mm_extract_epi16
  // CIR %{{.*}} = cir.vec.extract %{{.*}}[%{{.*}} : {{!u32i|!u64i}}] : !cir.vector<!s16i x 8>
  // CIR %{{.*}} = cir.cast integral %{{.*}} : !u16i -> !s32i

  // LLVM-LABEL: test_mm_extract_epi16
  // LLVM: extractelement <8 x i16> %{{.*}}, {{i32|i64}} 1
  // LLVM: zext i16 %{{.*}} to i32

  // OGCG-LABEL: test_mm_extract_epi16
  // OGCG: extractelement <8 x i16> %{{.*}}, {{i32|i64}} 1
  // OGCG: zext i16 %{{.*}} to i32
  return _mm_extract_epi16(A, 1);
}

void test_mm_clflush(void* A) {
  // CIR-LABEL: test_mm_clflush
  // LLVM-LABEL: test_mm_clflush
  // OGCG-LABEL: test_mm_clflush
  _mm_clflush(A);
  // CIR: {{%.*}} = cir.call_llvm_intrinsic "x86.sse2.clflush" {{%.*}} : (!cir.ptr<!void>) -> !void
  // LLVM: call void @llvm.x86.sse2.clflush(ptr {{%.*}})
  // OGCG: call void @llvm.x86.sse2.clflush(ptr {{%.*}})
}

void test_mm_lfence(void) {
  // CIR-LABEL: test_mm_lfence
  // LLVM-LABEL: test_mm_lfence
  // OGCG-LABEL: test_mm_lfence
  _mm_lfence();
  // CIR: {{%.*}} = cir.call_llvm_intrinsic "x86.sse2.lfence" : () -> !void
  // LLVM: call void @llvm.x86.sse2.lfence()
  // OGCG: call void @llvm.x86.sse2.lfence()
}

void test_mm_mfence(void) {
  // CIR-LABEL: test_mm_mfence
  // LLVM-LABEL: test_mm_mfence
  // OGCG-LABEL: test_mm_mfence
  _mm_mfence();
  // CIR: {{%.*}} = cir.call_llvm_intrinsic "x86.sse2.mfence" : () -> !void
  // LLVM: call void @llvm.x86.sse2.mfence()
  // OGCG: call void @llvm.x86.sse2.mfence()
}

void test_mm_pause(void) {
  // CIR-LABEL: test_mm_pause
  // LLVM-LABEL: test_mm_pause
  // OGCG-LABEL: test_mm_pause
  _mm_pause();
  // CIR: {{%.*}} = cir.call_llvm_intrinsic "x86.sse2.pause" : () -> !void
  // LLVM: call void @llvm.x86.sse2.pause()
  // OGCG: call void @llvm.x86.sse2.pause()
}

__m128i test_mm_shufflelo_epi16(__m128i A) {
  // CIR-LABEL: _mm_shufflelo_epi16
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !s16i>) [#cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i] : !cir.vector<8 x !s16i>

  // LLVM-LABEL: test_mm_shufflelo_epi16
  // LLVM: shufflevector <8 x i16> %{{.*}}, <8 x i16> poison, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 4, i32 5, i32 6, i32 7>

  // OGCG-LABEL: test_mm_shufflelo_epi16
  // OGCG: shufflevector <8 x i16> %{{.*}}, <8 x i16> poison, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 4, i32 5, i32 6, i32 7>
  return _mm_shufflelo_epi16(A, 0);
}

__m128i test_mm_shufflehi_epi16(__m128i A) {
  // CIR-LABEL: _mm_shufflehi_epi16
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !s16i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<4> : !s32i, #cir.int<4> : !s32i, #cir.int<4> : !s32i, #cir.int<4> : !s32i] : !cir.vector<8 x !s16i>

  // LLVM-LABEL: test_mm_shufflehi_epi16
  // LLVM: shufflevector <8 x i16> %{{.*}}, <8 x i16> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 4, i32 4, i32 4>

  // OGCG-LABEL: test_mm_shufflehi_epi16
  // OGCG: shufflevector <8 x i16> %{{.*}}, <8 x i16> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 4, i32 4, i32 4>
  return _mm_shufflehi_epi16(A, 0);
}

__m128d test_mm_shuffle_pd(__m128d A, __m128d B) {
  // CIR-LABEL: test_mm_shuffle_pd
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !cir.double>) [#cir.int<1> : !s32i, #cir.int<2> : !s32i] : !cir.vector<2 x !cir.double>

  // LLVM-LABEL: test_mm_shuffle_pd
  // LLVM: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 1, i32 2>

  // OGCG-LABEL: test_mm_shuffle_pd
  // OGCG: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 1, i32 2>
  return _mm_shuffle_pd(A, B, 1);
}

__m128i test_mm_shuffle_epi32(__m128i A) {
	// CIR-LABEL: test_mm_shuffle_epi32
	// CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}}: !cir.vector<4 x !s32i>) [#cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<0> : !s32i, #cir.int<1> : !s32i] : !cir.vector<4 x !s32i>

    // LLVM-LABEL: test_mm_shuffle_epi32
	// LLVM: shufflevector <4 x i32> %{{.*}}, <4 x i32> poison, <4 x i32> <i32 2, i32 3, i32 0, i32 1>

	// OGCG-LABEL: test_mm_shuffle_epi32
    // OGCG: shufflevector <4 x i32> %{{.*}}, <4 x i32> poison, <4 x i32> <i32 2, i32 3, i32 0, i32 1>
    return _mm_shuffle_epi32(A, 0x4E);
}

__m128i test_mm_mul_epu32(__m128i A, __m128i B) {
  // CIR-LABEL: _mm_mul_epu32
  // CIR: [[BC_A:%.*]] = cir.cast bitcast %{{.*}} : {{.*}} -> !cir.vector<2 x !s64i>
  // CIR: [[BC_B:%.*]] = cir.cast bitcast %{{.*}} : {{.*}} -> !cir.vector<2 x !s64i>
  // CIR: [[MASK_SCALAR:%.*]] = cir.const #cir.int<4294967295> : !s64i
  // CIR: [[MASK_VEC:%.*]] = cir.vec.splat [[MASK_SCALAR]] : !s64i, !cir.vector<2 x !s64i>
  // CIR: [[AND_A:%.*]] = cir.binop(and, [[BC_A]], [[MASK_VEC]])
  // CIR: [[AND_B:%.*]] = cir.binop(and, [[BC_B]], [[MASK_VEC]])
  // CIR: [[MUL:%.*]]   = cir.binop(mul, [[AND_A]], [[AND_B]])

  // LLVM-LABEL: _mm_mul_epu32
  // LLVM: and <2 x i64> %{{.*}}, splat (i64 4294967295)
  // LLVM: and <2 x i64> %{{.*}}, splat (i64 4294967295)
  // LLVM: mul <2 x i64> %{{.*}}, %{{.*}}

  // OGCG-LABEL: _mm_mul_epu32
  // OGCG: and <2 x i64> %{{.*}}, splat (i64 4294967295)
  // OGCG: and <2 x i64> %{{.*}}, splat (i64 4294967295)
  // OGCG: mul <2 x i64> %{{.*}}, %{{.*}}

  return _mm_mul_epu32(A, B);
}
