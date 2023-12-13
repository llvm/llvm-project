// RUN: %clang_cc1 -triple x86_64 -ffreestanding -target-cpu cannonlake -emit-llvm < %s | FileCheck %s

#include <immintrin.h>

int main(int argc, char **argv) {
  // CHECK-LABEL: @main
  // CHECK: @llvm.masked.load.v4i64.p0
  __m256i ptrs = _mm256_maskz_loadu_epi64(0, argv);
  return 0;
}
