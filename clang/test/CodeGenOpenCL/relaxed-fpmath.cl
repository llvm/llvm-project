// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s -check-prefix=NORMAL
// RUN: %clang_cc1 %s -emit-llvm -cl-fast-relaxed-math -o - | FileCheck %s -check-prefix=FAST
// RUN: %clang_cc1 %s -emit-llvm -cl-finite-math-only -o - | FileCheck %s -check-prefix=FINITE
// RUN: %clang_cc1 %s -emit-llvm -cl-unsafe-math-optimizations -o - | FileCheck %s -check-prefix=UNSAFE
// RUN: %clang_cc1 %s -emit-llvm -cl-mad-enable -o - | FileCheck %s -check-prefix=MAD
// RUN: %clang_cc1 %s -emit-llvm -cl-no-signed-zeros -o - | FileCheck %s -check-prefix=NOSIGNED

float spscalardiv(float a, float b) {
  // CHECK: @spscalardiv(

  // NORMAL: fdiv float
  // FAST: fdiv fast float
  // FINITE: fdiv nnan ninf float
  // UNSAFE: fdiv reassoc nsz arcp afn float
  // MAD: fdiv float
  // NOSIGNED: fdiv nsz float
  return a / b;
}
// CHECK: attributes

// NORMAL: "less-precise-fpmad"="false"
// NORMAL: "no-infs-fp-math"="false"
// NORMAL: "no-nans-fp-math"="false"
// NORMAL: "no-signed-zeros-fp-math"="false"
// NORMAL: "unsafe-fp-math"="false"

// FAST: "less-precise-fpmad"="true"
// FAST: "no-infs-fp-math"="true"
// FAST: "no-nans-fp-math"="true"
// FAST: "no-signed-zeros-fp-math"="true"
// FAST: "unsafe-fp-math"="true"

// FINITE: "less-precise-fpmad"="false"
// FINITE: "no-infs-fp-math"="true"
// FINITE: "no-nans-fp-math"="true"
// FINITE: "no-signed-zeros-fp-math"="false"
// FINITE: "unsafe-fp-math"="false"

// UNSAFE: "less-precise-fpmad"="true"
// UNSAFE: "no-infs-fp-math"="false"
// UNSAFE: "no-nans-fp-math"="false"
// UNSAFE: "no-signed-zeros-fp-math"="true"
// UNSAFE: "unsafe-fp-math"="true"

// MAD: "less-precise-fpmad"="true"
// MAD: "no-infs-fp-math"="false"
// MAD: "no-nans-fp-math"="false"
// MAD: "no-signed-zeros-fp-math"="false"
// MAD: "unsafe-fp-math"="false"

// NOSIGNED: "less-precise-fpmad"="false"
// NOSIGNED: "no-infs-fp-math"="false"
// NOSIGNED: "no-nans-fp-math"="false"
// NOSIGNED: "no-signed-zeros-fp-math"="true"
// NOSIGNED: "unsafe-fp-math"="false"
