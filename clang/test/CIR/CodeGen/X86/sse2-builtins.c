// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse2 -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefixes=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse2 -fno-signed-char -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefixes=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse2 -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse2 -fno-signed-char -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=OGCG

// This test mimics clang/test/CodeGen/X86/sse2-builtins.c, which eventually
// CIR shall be able to support fully.

#include <immintrin.h>

__m128d test_mm_undefined_pd(void) {
  // CIR-X64-LABEL: _mm_undefined_pd
  // CIR-X64: %{{.*}} = cir.const #cir.zero : !cir.vector<2 x !cir.double>
  // CIR-X64: cir.return %{{.*}} : !cir.vector<2 x !cir.double>
  // LLVM-X64-LABEL: test_mm_undefined_pd
  // LLVM-X64: store <2 x double> zeroinitializer, ptr %[[A:.*]], align 16
  // LLVM-X64: %{{.*}} = load <2 x double>, ptr %[[A]], align 16
  // LLVM-X64: ret <2 x double> %{{.*}}
  return _mm_undefined_pd();
}

__m128i test_mm_undefined_si128(void) {
  // CIR-LABEL: _mm_undefined_si128
  // CIR-CHECK: %[[A:.*]] = cir.const #cir.zero : !cir.vector<2 x !cir.double>
  // CIR-CHECK: %{{.*}} = cir.cast bitcast %[[A]] : !cir.vector<2 x !cir.double> -> !cir.vector<2 x !s64i>
  // CIR-CHECK: cir.return %{{.*}} : !cir.vector<2 x !s64i>

  // LLVM-CHECK-LABEL: test_mm_undefined_si128
  // LLVM-CHECK: store <2 x i64> zeroinitializer, ptr %[[A:.*]], align 16
  // LLVM-CHECK: %{{.*}} = load <2 x i64>, ptr %[[A]], align 16
  // LLVM-CHECK: ret <2 x i64> %{{.*}}
  return _mm_undefined_si128();
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
