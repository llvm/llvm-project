// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-apple-darwin -target-feature +clwb  -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <x86intrin.h>

void test_mm_clwb(const void *__m) {
  //CHECK-LABEL: @test_mm_clwb
  //CHECK: @llvm.x86.clwb
  _mm_clwb(__m);
}
