// RUN: %clang_cc1 -ffreestanding -triple x86_64-unknown-linux -Wno-implicit-function-declaration -target-feature +sse2 -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -ffreestanding -triple x86_64-unknown-linux -Wno-implicit-function-declaration -target-feature +sse2 -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -ffreestanding -triple x86_64-unknown-linux -Wno-implicit-function-declaration -target-feature -sse2 -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -ffreestanding -triple x86_64-unknown-linux -Wno-implicit-function-declaration -target-feature -sse2 -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// This test mimics clang/test/CodeGen/X86/pause.c, which eventually
// CIR shall be able to support fully.

#include <x86intrin.h>

void test_mm_pause(void) {
  // CIR-LABEL: test_mm_pause
  // LLVM-LABEL: test_mm_pause
  _mm_pause();
  // CIR: {{%.*}} = cir.llvm.intrinsic "x86.sse2.pause" : () -> !void
  // LLVM: call void @llvm.x86.sse2.pause()
}
