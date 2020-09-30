// RUN: %clang -target aarch64 -emit-llvm -S %s -o - | FileCheck %s
// RUN: %clang -target lanai -emit-llvm -S %s -o - | FileCheck %s
// RUN: %clang -target riscv64 -emit-llvm -S %s -o - | FileCheck %s
// RUN: %clang -target x86_64 -emit-llvm -S %s -o - | FileCheck %s

// Run a variety of targets to ensure there's no target-based difference.

// The builtin always produces a 64-bit (double).
// An SNaN with no payload is formed by setting the bit after the
// the quiet bit (MSB of the significand).

// CHECK: float 0x7FF8000000000000, float 0x7FF4000000000000
// CHECK: double 0x7FF8000000000000, double 0x7FF4000000000000

float f[] = {
  __builtin_nan(""),
  __builtin_nans(""),
};

double d[] = {
  __builtin_nan(""),
  __builtin_nans(""),
};
