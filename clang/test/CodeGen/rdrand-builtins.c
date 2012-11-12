// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-feature +rdrnd -emit-llvm -S -emit-llvm -o - %s | FileCheck %s

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <immintrin.h>

int rdrand16(unsigned short *p) {
  return _rdrand16_step(p);
// CHECK: @rdrand16
// CHECK: call { i16, i32 } @llvm.x86.rdrand.16
// CHECK: store i16
}

int rdrand32(unsigned *p) {
  return _rdrand32_step(p);
// CHECK: @rdrand32
// CHECK: call { i32, i32 } @llvm.x86.rdrand.32
// CHECK: store i32
}

int rdrand64(unsigned long long *p) {
  return _rdrand64_step(p);
// CHECK: @rdrand64
// CHECK: call { i64, i32 } @llvm.x86.rdrand.64
// CHECK: store i64
}
