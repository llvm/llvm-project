// RUN: %clang_cc1 -triple arm-none-eabi -ffreestanding -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64 -ffreestanding -emit-llvm -o - %s | FileCheck %s

extern struct T {
  int b0 : 8;
  int b1 : 24;
  int b2 : 1;
} g;

int func(void) {
  return g.b1;
}

// CHECK: @g = external global %struct.T, align 4
// CHECK: %{{.*}} = load i64, i64* bitcast (%struct.T* @g to i64*), align 4
