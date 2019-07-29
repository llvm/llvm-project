// Bug: https://bugs.llvm.org/show_bug.cgi?id=42668
// REQUIRES: arm-registered-target
// RUN: %clang --target=arm-arm-none-eabi -march=armv8-a -S -emit-llvm -Os -o - %s | FileCheck --check-prefixes=CHECK,A8 %s
// RUN: %clang --target=arm-linux-androideabi -march=armv8-a -S -emit-llvm -Os -o - %s | FileCheck --check-prefixes=CHECK,A16 %s
// CHECK: [[E:%[A-z0-9]+]] = tail call i8* @__cxa_allocate_exception
// CHECK-NEXT: [[BC:%[A-z0-9]+]] = bitcast i8* [[E]] to <2 x i64>*
// A8-NEXT: store <2 x i64> <i64 1, i64 2>, <2 x i64>* [[BC]], align 8
// A16-NEXT: store <2 x i64> <i64 1, i64 2>, <2 x i64>* [[BC]], align 16
#include <arm_neon.h>

int main(void) {
  try {
    throw vld1q_u64(((const uint64_t[2]){1, 2}));
  } catch (uint64x2_t exc) {
    return 0;
  }
  return 1;
}

