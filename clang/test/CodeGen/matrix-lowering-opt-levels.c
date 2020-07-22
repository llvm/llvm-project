// RUN: %clang -O0 -fenable-matrix -S -emit-llvm %s -o - | FileCheck  %s
// RUN: %clang -O1 -fenable-matrix -S -emit-llvm %s -o - | FileCheck  %s
// RUN: %clang -O2 -fenable-matrix -S -emit-llvm %s -o - | FileCheck  %s
// RUN: %clang -O3 -fenable-matrix -S -emit-llvm %s -o - | FileCheck  %s
// RUN: %clang -Ofast -fenable-matrix -S -emit-llvm %s -o - | FileCheck  %s
// RUN: %clang -Os -fenable-matrix -S -emit-llvm %s -o - | FileCheck  %s
// RUN: %clang -Oz -fenable-matrix -S -emit-llvm %s -o - | FileCheck  %s

// RUN: %clang -O0 -fenable-matrix -fexperimental-new-pass-manager -S -emit-llvm %s -o - | FileCheck  %s
// RUN: %clang -O1 -fenable-matrix -fexperimental-new-pass-manager -S -emit-llvm %s -o - | FileCheck  %s
// RUN: %clang -O2 -fenable-matrix -fexperimental-new-pass-manager -S -emit-llvm %s -o - | FileCheck  %s
// RUN: %clang -O3 -fenable-matrix -fexperimental-new-pass-manager -S -emit-llvm %s -o - | FileCheck  %s
// RUN: %clang -Ofast -fenable-matrix -fexperimental-new-pass-manager -S -emit-llvm %s -o - | FileCheck  %s
// RUN: %clang -Os -fenable-matrix -fexperimental-new-pass-manager -S -emit-llvm %s -o - | FileCheck  %s
// RUN: %clang -Oz -fenable-matrix -fexperimental-new-pass-manager -S -emit-llvm %s -o - | FileCheck  %s

// Smoke test that the matrix intrinsics are lowered at any optimisation level.

typedef float m4x4_t __attribute__((matrix_type(4, 4)));

m4x4_t f(m4x4_t a, m4x4_t b, m4x4_t c) {
  //
  // CHECK-LABEL: f(
  // CHECK-NOT:     @llvm.matrix
  // CHECK:       }
  //
  return a + b * c;
}
