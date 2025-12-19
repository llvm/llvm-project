// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse -fno-signed-char -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=OGCG

// This test mimics clang/test/CodeGen/X86/sse-builtins.c, which eventually
// CIR shall be able to support fully.

#include <immintrin.h>

void test_mm_setcsr(unsigned int A) {
  // CIR-LABEL: test_mm_setcsr
  // CIR: cir.store {{.*}}, {{.*}} : !u32i
  // CIR: cir.call_llvm_intrinsic "x86.sse.ldmxcsr" {{.*}} : (!cir.ptr<!u32i>) -> !void

  // LLVM-LABEL: test_mm_setcsr
  // LLVM: store i32
  // LLVM: call void @llvm.x86.sse.ldmxcsr(ptr {{.*}})

  // OGCG-LABEL: test_mm_setcsr
  // OGCG: store i32
  // OGCG: call void @llvm.x86.sse.ldmxcsr(ptr {{.*}})
  _mm_setcsr(A);
}

unsigned int test_mm_getcsr(void) {
  // CIR-LABEL: test_mm_getcsr
  // CIR: cir.call_llvm_intrinsic "x86.sse.stmxcsr" %{{.*}} : (!cir.ptr<!u32i>) -> !void
  // CIR: cir.load {{.*}} : !cir.ptr<!u32i>, !u32i

  // LLVM-LABEL: test_mm_getcsr
  // LLVM: call void @llvm.x86.sse.stmxcsr(ptr %{{.*}})
  // LLVM: load i32

  // OGCG-LABEL: test_mm_getcsr
  // OGCG: call void @llvm.x86.sse.stmxcsr(ptr %{{.*}})
  // OGCG: load i32
  return _mm_getcsr();
}

void test_mm_sfence(void) {
  // CIR-LABEL: test_mm_sfence
  // LLVM-LABEL: test_mm_sfence
  // OGCG-LABEL: test_mm_sfence
  _mm_sfence();
  // CIR: {{%.*}} = cir.call_llvm_intrinsic "x86.sse.sfence" : () -> !void
  // LLVM: call void @llvm.x86.sse.sfence()
  // OGCG: call void @llvm.x86.sse.sfence()
}

void test_mm_prefetch(char const* p) {
  // CIR-LABEL: test_mm_prefetch
  // LLVM-LABEL: test_mm_prefetch
  // OGCG-LABEL: test_mm_prefetch
  _mm_prefetch(p, 0);
  // CIR: cir.prefetch read locality(0) %{{.*}} : !cir.ptr<!void>
  // LLVM: call void @llvm.prefetch.p0(ptr {{.*}}, i32 0, i32 0, i32 1)
  // OGCG: call void @llvm.prefetch.p0(ptr {{.*}}, i32 0, i32 0, i32 1)
}

void test_mm_prefetch_local(char const* p) {
  // CIR-LABEL: test_mm_prefetch_local
  // LLVM-LABEL: test_mm_prefetch_local
  // OGCG-LABEL: test_mm_prefetch_local
  _mm_prefetch(p, 3);
  // CIR: cir.prefetch read locality(3) %{{.*}} : !cir.ptr<!void>
  // LLVM: call void @llvm.prefetch.p0(ptr {{.*}}, i32 0, i32 3, i32 1)
  // OGCG: call void @llvm.prefetch.p0(ptr {{.*}}, i32 0, i32 3, i32 1)
}

void test_mm_prefetch_write(char const* p) {
  // CIR-LABEL: test_mm_prefetch_write
  // LLVM-LABEL: test_mm_prefetch_write
  // OGCG-LABEL: test_mm_prefetch_write
  _mm_prefetch(p, 7);
  // CIR: cir.prefetch write locality(3) %{{.*}} : !cir.ptr<!void>
  // LLVM: call void @llvm.prefetch.p0(ptr {{.*}}, i32 1, i32 3, i32 1)
  // OGCG: call void @llvm.prefetch.p0(ptr {{.*}}, i32 1, i32 3, i32 1)
}

__m128 test_mm_undefined_ps(void) {
  // CIR-LABEL: _mm_undefined_ps
  // CIR: %[[A:.*]] = cir.const #cir.zero : !cir.vector<2 x !cir.double>
  // CIR: %{{.*}} = cir.cast bitcast %[[A]] : !cir.vector<2 x !cir.double> -> !cir.vector<4 x !cir.float>
  // CIR: cir.return %{{.*}} : !cir.vector<4 x !cir.float>

  // LLVM-LABEL: test_mm_undefined_ps
  // LLVM: store <4 x float> zeroinitializer, ptr %[[A:.*]], align 16
  // LLVM: %{{.*}} = load <4 x float>, ptr %[[A]], align 16
  // LLVM: ret <4 x float> %{{.*}}

  // OGCG-LABEL: test_mm_undefined_ps
  // OGCG: ret <4 x float> zeroinitializer
  return _mm_undefined_ps();
}

__m128 test_mm_shuffle_ps(__m128 A, __m128 B) {
  // CIR-LABEL: _mm_shuffle_ps
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float>) [#cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<4> : !s32i, #cir.int<4> : !s32i] : !cir.vector<4 x !cir.float>

  // LLVM-LABEL: test_mm_shuffle_ps
  // LLVM: shufflevector <4 x float> {{.*}}, <4 x float> {{.*}}, <4 x i32> <i32 0, i32 0, i32 4, i32 4>

  // OGCG-LABEL: test_mm_shuffle_ps
  // OGCG: shufflevector <4 x float> {{.*}}, <4 x float> {{.*}}, <4 x i32> <i32 0, i32 0, i32 4, i32 4>
  return _mm_shuffle_ps(A, B, 0);
}
