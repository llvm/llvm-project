// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse -fno-signed-char -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=OGCG

// This test mimics clang/test/CodeGen/X86/sse-builtins.c, which eventually
// CIR shall be able to support fully.

#include <immintrin.h>


void test_mm_prefetch(char const* p) {
  // CIR-LABEL: test_mm_prefetch
  // LLVM-LABEL: test_mm_prefetch
  _mm_prefetch(p, 0);
  // CIR: cir.prefetch(%{{.*}} : !cir.ptr<!void>) locality(0) read
  // LLVM: call void @llvm.prefetch.p0(ptr {{.*}}, i32 0, i32 0, i32 1)
}

void test_mm_sfence(void) {
  // CIR-LABEL: test_mm_sfence
  // LLVM-LABEL: test_mm_sfence
  _mm_sfence();
  // CIR: {{%.*}} = cir.llvm.intrinsic "x86.sse.sfence" : () -> !void
  // LLVM: call void @llvm.x86.sse.sfence()
}

__m128 test_mm_undefined_ps(void) {
  // CIR-LABEL: _mm_undefined_ps
  // CIR: %[[A:.*]] = cir.const #cir.zero : !cir.vector<!cir.double x 2>
  // CIR: %{{.*}} = cir.cast bitcast %[[A]] : !cir.vector<!cir.double x 2> -> !cir.vector<!cir.float x 4>
  // CIR: cir.return %{{.*}} : !cir.vector<!cir.float x 4>

  // LLVM-LABEL: test_mm_undefined_ps
  // LLVM: store <4 x float> zeroinitializer, ptr %[[A:.*]], align 16
  // LLVM: %{{.*}} = load <4 x float>, ptr %[[A]], align 16
  // LLVM: ret <4 x float> %{{.*}}
  return _mm_undefined_ps();
}

void test_mm_setcsr(unsigned int A) {
  // CIR-LABEL: test_mm_setcsr
  // CIR: cir.store {{.*}}, {{.*}} : !u32i
  // CIR: cir.llvm.intrinsic "x86.sse.ldmxcsr" {{.*}} : (!cir.ptr<!u32i>) -> !void

  // LLVM-LABEL: test_mm_setcsr 
  // LLVM: store i32
  // LLVM: call void @llvm.x86.sse.ldmxcsr(ptr {{.*}})
  _mm_setcsr(A);
}

unsigned int test_mm_getcsr(void) {
  // CIR-LABEL: test_mm_getcsr
  // CIR: cir.llvm.intrinsic "x86.sse.stmxcsr" %{{.*}} : (!cir.ptr<!u32i>) -> !void
  // CIR: cir.load {{.*}} : !cir.ptr<!u32i>, !u32i

  // LLVM-LABEL: test_mm_getcsr
  // LLVM: call void @llvm.x86.sse.stmxcsr(ptr %{{.*}})
  // LLVM: load i32
  return _mm_getcsr();
}

__m128 test_mm_shuffle_ps(__m128 A, __m128 B) {
  // CIR-LABEL: _mm_shuffle_ps
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!cir.float x 4>) [#cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<4> : !s32i, #cir.int<4> : !s32i] : !cir.vector<!cir.float x 4>

  // CHECK-LABEL: test_mm_shuffle_ps
  // CHECK: shufflevector <4 x float> {{.*}}, <4 x float> {{.*}}, <4 x i32> <i32 0, i32 0, i32 4, i32 4>

  // OGCG-LABEL: test_mm_shuffle_ps
  // OGCG: shufflevector <4 x float> {{.*}}, <4 x float> {{.*}}, <4 x i32> <i32 0, i32 0, i32 4, i32 4>
  return _mm_shuffle_ps(A, B, 0);
}