// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s -check-prefix=NORMAL
// RUN: %clang_cc1 %s -emit-llvm -cl-fast-relaxed-math -o - | FileCheck %s -check-prefix=FAST
// RUN: %clang_cc1 %s -emit-llvm -cl-finite-math-only -o - | FileCheck %s -check-prefix=FINITE
// RUN: %clang_cc1 %s -emit-llvm -cl-unsafe-math-optimizations -o - | FileCheck %s -check-prefix=UNSAFE
// RUN: %clang_cc1 %s -emit-llvm -cl-mad-enable -o - | FileCheck %s -check-prefix=MAD
// RUN: %clang_cc1 %s -emit-llvm -cl-no-signed-zeros -o - | FileCheck %s -check-prefix=NOSIGNED

// Check the fp options are correct with PCH.
// RUN: %clang_cc1 %s -DGEN_PCH=1 -finclude-default-header -triple spir-unknown-unknown -emit-pch -o %t.pch
// RUN: %clang_cc1 %s -include-pch %t.pch -fno-validate-pch -emit-llvm -o - | FileCheck %s -check-prefix=NORMAL
// RUN: %clang_cc1 %s -include-pch %t.pch -fno-validate-pch -emit-llvm -cl-fast-relaxed-math -o - | FileCheck %s -check-prefix=FAST
// RUN: %clang_cc1 %s -include-pch %t.pch -fno-validate-pch -emit-llvm -cl-finite-math-only -o - | FileCheck %s -check-prefix=FINITE
// RUN: %clang_cc1 %s -include-pch %t.pch -fno-validate-pch -emit-llvm -cl-unsafe-math-optimizations -o - | FileCheck %s -check-prefix=UNSAFE
// RUN: %clang_cc1 %s -include-pch %t.pch -fno-validate-pch -emit-llvm -cl-mad-enable -o - | FileCheck %s -check-prefix=MAD
// RUN: %clang_cc1 %s -include-pch %t.pch -fno-validate-pch -emit-llvm -cl-no-signed-zeros -o - | FileCheck %s -check-prefix=NOSIGNED

#if !GEN_PCH

float spscalardiv(float a, float b) {
  // CHECK: @spscalardiv(

  // NORMAL: fdiv float
  // FAST: fdiv fast float
  // FINITE: fdiv nnan ninf float
  // UNSAFE: fdiv reassoc nsz arcp contract afn float
  // MAD: fdiv float
  // NOSIGNED: fdiv nsz float
  return a / b;
}
// CHECK: attributes

// NORMAL-NOT: "less-precise-fpmad"
// NORMAL-NOT: "no-infs-fp-math"
// NORMAL-NOT: "no-nans-fp-math"
// NORMAL-NOT: "no-signed-zeros-fp-math"
// NORMAL-NOT: "unsafe-fp-math"

// FAST: "less-precise-fpmad"="true"
// FAST: "no-infs-fp-math"="true"
// FAST: "no-nans-fp-math"="true"
// FAST: "no-signed-zeros-fp-math"="true"
// FAST: "unsafe-fp-math"="true"

// FINITE-NOT: "less-precise-fpmad"
// FINITE: "no-infs-fp-math"="true"
// FINITE: "no-nans-fp-math"="true"
// FINITE-NOT: "no-signed-zeros-fp-math"
// FINITE-NOT: "unsafe-fp-math"

// UNSAFE: "less-precise-fpmad"="true"
// UNSAFE-NOT: "no-infs-fp-math"
// UNSAFE-NOT: "no-nans-fp-math"
// UNSAFE: "no-signed-zeros-fp-math"="true"
// UNSAFE: "unsafe-fp-math"="true"

// MAD: "less-precise-fpmad"="true"
// MAD-NOT: "no-infs-fp-math"
// MAD-NOT: "no-nans-fp-math"
// MAD-NOT: "no-signed-zeros-fp-math"
// MAD-NOT: "unsafe-fp-math"

// NOSIGNED-NOT: "less-precise-fpmad"
// NOSIGNED-NOT: "no-infs-fp-math"
// NOSIGNED-NOT: "no-nans-fp-math"
// NOSIGNED: "no-signed-zeros-fp-math"="true"
// NOSIGNED-NOT: "unsafe-fp-math"

#else
// Undefine this to avoid putting it in the PCH.
#undef GEN_PCH
#endif
