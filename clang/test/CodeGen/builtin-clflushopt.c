// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +clflushopt  -emit-llvm -o - -Werror | FileCheck %s
#define __MM_MALLOC_H

#include <immintrin.h>
void test_mm_clflushopt(char * __m) {
  //CHECK-LABLE: @test_mm_clflushopt
  //CHECK: @llvm.x86.clflushopt
  _mm_clflushopt(__m);
}
