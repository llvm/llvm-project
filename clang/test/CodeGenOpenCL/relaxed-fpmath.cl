// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s -check-prefix=NORMAL
// RUN: %clang_cc1 %s -emit-llvm -cl-fast-relaxed-math -menable-no-nans -menable-no-infs -mreassociate -freciprocal-math -fapprox-func -fno-signed-zeros -o - | FileCheck %s -check-prefix=FAST
// RUN: %clang_cc1 %s -emit-llvm -menable-no-infs -menable-no-nans -cl-finite-math-only -o - | FileCheck %s -check-prefix=FINITE
// RUN: %clang_cc1 %s -emit-llvm -cl-unsafe-math-optimizations -o - | FileCheck %s -check-prefix=UNSAFE
// RUN: %clang_cc1 %s -emit-llvm -cl-mad-enable -o - | FileCheck %s -check-prefix=MAD
// RUN: %clang_cc1 %s -emit-llvm -cl-no-signed-zeros -o - | FileCheck %s -check-prefix=NOSIGNED

// -cl-fast-relaxed-math alone implies -ffast-math at the cc1 level, so both
// -menable-no-nans and -menable-no-infs are implied.
// RUN: %clang_cc1 %s -emit-llvm -cl-fast-relaxed-math -o - | FileCheck %s -check-prefix=FAST
// A caller (e.g. a cc1-direct invocation) can cancel just one side of the
// implied fast-math NaN/Inf assumptions via the negation flags.
// RUN: %clang_cc1 %s -emit-llvm -cl-fast-relaxed-math -mno-enable-no-nans -o - | FileCheck %s -check-prefix=NONAN
// RUN: %clang_cc1 %s -emit-llvm -cl-fast-relaxed-math -mno-enable-no-infs -o - | FileCheck %s -check-prefix=NOINF

// Check the fp options are correct with PCH.
// RUN: %clang_cc1 %s -DGEN_PCH=1 -finclude-default-header -triple spir-unknown-unknown -emit-pch -o %t.pch
// RUN: %clang_cc1 %s -include-pch %t.pch -fno-validate-pch -emit-llvm -o - | FileCheck %s -check-prefix=NORMAL
// RUN: %clang_cc1 %s -include-pch %t.pch -fno-validate-pch -emit-llvm -cl-fast-relaxed-math -menable-no-nans -menable-no-infs -mreassociate -freciprocal-math -fapprox-func -fno-signed-zeros -o - | FileCheck %s -check-prefix=FAST
// RUN: %clang_cc1 %s -include-pch %t.pch -fno-validate-pch -emit-llvm -menable-no-infs -menable-no-nans -cl-finite-math-only -o - | FileCheck %s -check-prefix=FINITE
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
  // NONAN: fdiv reassoc ninf nsz arcp contract afn float
  // NOINF: fdiv reassoc nnan nsz arcp contract afn float
  return a / b;
}
// CHECK: attributes

// NORMAL-NOT: "less-precise-fpmad"
// NORMAL-NOT: "no-signed-zeros-fp-math"

// FAST: "less-precise-fpmad"="true"
// FAST: "no-signed-zeros-fp-math"="true"

// FINITE-NOT: "less-precise-fpmad"
// FINITE-NOT: "no-signed-zeros-fp-math"

// UNSAFE: "less-precise-fpmad"="true"
// UNSAFE: "no-signed-zeros-fp-math"="true"

// MAD: "less-precise-fpmad"="true"
// MAD-NOT: "no-signed-zeros-fp-math"

// NOSIGNED-NOT: "less-precise-fpmad"
// NOSIGNED: "no-signed-zeros-fp-math"="true"

#else
// Undefine this to avoid putting it in the PCH.
#undef GEN_PCH
#endif
