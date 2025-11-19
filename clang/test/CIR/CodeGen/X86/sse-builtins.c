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
