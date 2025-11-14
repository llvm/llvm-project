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
