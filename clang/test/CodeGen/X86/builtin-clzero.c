// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-apple-darwin -target-feature +clzero  -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <x86intrin.h>

void test_mm_clzero(void * __m) {
  //CHECK-LABEL: @test_mm_clzero
  //CHECK: @llvm.x86.clzero
  _mm_clzero(__m);
}
