// RUN: %clang -target aarch64 -emit-llvm -S %s -o - | FileCheck %s
// RUN: %clang -target lanai -emit-llvm -S %s -o - | FileCheck %s
// RUN: %clang -target riscv64 -emit-llvm -S %s -o - | FileCheck %s
// RUN: %clang -target x86_64 -emit-llvm -S %s -o - | FileCheck %s

// Run a variety of targets to ensure there's no target-based difference.

// An SNaN with no payload is formed by setting the bit after the
// the quiet bit (MSB of the significand).

// CHECK: float 0x7FF8000000000000, float 0x7FF4000000000000

float f[] = {
  __builtin_nanf(""),
  __builtin_nansf(""),
};


// Doubles are created and converted to floats.
// Converting (truncating) to float quiets the NaN (sets the MSB
// of the significand) and raises the APFloat invalidOp exception
// but that should not cause a compilation error in the default
// (ignore FP exceptions) mode.

// CHECK: float 0x7FF8000000000000, float 0x7FFC000000000000

float converted_to_float[] = {
  __builtin_nan(""),
  __builtin_nans(""),
};

// CHECK: double 0x7FF8000000000000, double 0x7FF4000000000000

double d[] = {
  __builtin_nan(""),
  __builtin_nans(""),
};
