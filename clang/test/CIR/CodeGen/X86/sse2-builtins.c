// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse2 -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefixes=CIR-CHECK --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse2 -fno-signed-char -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefixes=CIR-CHECK --input-file=%t.cir %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse2 -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM-CHECK --input-file=%t.ll %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse2 -fno-signed-char -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM-CHECK --input-file=%t.ll %s

// This test mimics clang/test/CodeGen/X86/sse2-builtins.c, which eventually
// CIR shall be able to support fully.

#include <immintrin.h>


void test_mm_clflush(void* A) {
  // CIR-LABEL: test_mm_clflush
  // LLVM-LABEL: teh
  _mm_clflush(A);
  // CIR-CHECK: {{%.*}} = cir.call_llvm_intrinsic "x86.sse2.clflush" {{%.*}} : (!cir.ptr<!void>) -> !void
  // LLVM-CHECK: call void @llvm.x86.sse2.clflush(ptr {{%.*}})
}

void test_mm_lfence(void) {
  // CIR-CHECK-LABEL: test_mm_lfence
  // LLVM-CHECK-LABEL: test_mm_lfence
  _mm_lfence();
  // CIR-CHECK: {{%.*}} = cir.call_llvm_intrinsic "x86.sse2.lfence" : () -> !void
  // LLVM-CHECK: call void @llvm.x86.sse2.lfence()
}

void test_mm_mfence(void) {
  // CIR-CHECK-LABEL: test_mm_mfence
  // LLVM-CHECK-LABEL: test_mm_mfence
  _mm_mfence();
  // CIR-CHECK: {{%.*}} = cir.call_llvm_intrinsic "x86.sse2.mfence" : () -> !void
  // LLVM-CHECK: call void @llvm.x86.sse2.mfence()
}

void test_mm_pause(void) {
  // CIR-LABEL: test_mm_pause
  // LLVM-LABEL: test_mm_pause
  _mm_pause();
  // CIR: {{%.*}} = cir.call_llvm_intrinsic "x86.sse2.pause" : () -> !void
  // LLVM: call void @llvm.x86.sse2.pause()
}
