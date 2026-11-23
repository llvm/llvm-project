// RUN: %clang -target aarch64 -emit-llvm -S %s -o - | FileCheck %s
// RUN: %clang -target lanai -emit-llvm -S %s -o - | FileCheck %s
// RUN: %clang -target riscv64 -emit-llvm -S %s -o - | FileCheck %s
// RUN: %clang -target x86_64 -emit-llvm -S %s -o - | FileCheck %s

// Run a variety of targets to ensure there's no target-based difference.

// An SNaN with no payload is formed by setting the bit after the
// the quiet bit (MSB of the significand).

// CHECK: float +qnan, float +snan(0x200000)

float f[] = {
  __builtin_nanf(""),
  __builtin_nansf(""),
};


// Doubles are created and converted to floats.
// Converting (truncating) to float quiets the NaN (sets the MSB
// of the significand) and raises the APFloat invalidOp exception
// but that should not cause a compilation error in the default
// (ignore FP exceptions) mode.

// CHECK: float +qnan, float +nan(0x200000)

float converted_to_float[] = {
  __builtin_nan(""),
  __builtin_nans(""),
};

// CHECK: double +qnan, double +snan(0x4000000000000)

double d[] = {
  __builtin_nan(""),
  __builtin_nans(""),
};
