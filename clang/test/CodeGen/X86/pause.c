// RUN: %clang_cc1 -ffreestanding %s -triple=i386-pc-win32 -target-feature -sse2 -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -ffreestanding %s -triple=i386-pc-win32 -target-feature +sse2 -emit-llvm -o - -Wall -Werror | FileCheck %s


#include <x86intrin.h>

void test_mm_pause() {
  // CHECK-LABEL: test_mm_pause
  // CHECK: call void @llvm.x86.sse2.pause()
  return _mm_pause();
}
