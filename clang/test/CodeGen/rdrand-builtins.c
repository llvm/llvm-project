// RUN: %clang_cc1 -ffreestanding -triple x86_64-unknown-unknown -target-feature +rdrnd -target-feature +rdseed -emit-llvm -o - %s | FileCheck %s


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

int rdseed16(unsigned short *p) {
  return _rdseed16_step(p);
// CHECK: @rdseed16
// CHECK: call { i16, i32 } @llvm.x86.rdseed.16
// CHECK: store i16
}

int rdseed32(unsigned *p) {
  return _rdseed32_step(p);
// CHECK: @rdseed32
// CHECK: call { i32, i32 } @llvm.x86.rdseed.32
// CHECK: store i32
}

int rdseed64(unsigned long long *p) {
  return _rdseed64_step(p);
// CHECK: @rdseed64
// CHECK: call { i64, i32 } @llvm.x86.rdseed.64
// CHECK: store i64
}
