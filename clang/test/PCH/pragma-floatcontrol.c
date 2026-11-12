// Test this without pch.
// RUN: %clang_cc1 -fexperimental-strict-floating-point  %s -include %s -verify -fsyntax-only -DSET
// RUN: %clang_cc1 -fexperimental-strict-floating-point  %s -include %s -verify -fsyntax-only -DPUSH
// RUN: %clang_cc1 -fexperimental-strict-floating-point  %s -include %s -verify -fsyntax-only -DPUSH_POP

// Test with pch.
// RUN: %clang_cc1 -fexperimental-strict-floating-point  %s -DSET -emit-pch -o %t
// RUN: %clang_cc1 -fexperimental-strict-floating-point  %s -DSET -include-pch %t -emit-llvm -o - | FileCheck --check-prefix=CHECK-EBSTRICT %s
// RUN: %clang_cc1 -fexperimental-strict-floating-point  %s -ffp-contract=on -DSET -emit-pch -o %t
// RUN: %clang_cc1 -fexperimental-strict-floating-point  %s -DSET -include-pch %t -emit-llvm -o - | FileCheck --check-prefix=CHECK-EBSTRICT %s
// RUN: %clang_cc1 -fexperimental-strict-floating-point  %s -menable-no-nans -DSET -emit-pch -o %t
// RUN: %clang_cc1 -fexperimental-strict-floating-point  %s -DSET -include-pch %t -emit-llvm -o - | FileCheck --check-prefix=CHECK-EBSTRICT %s
// RUN: %clang_cc1 -fexperimental-strict-floating-point  %s -frounding-math -DSET -emit-pch -o %t
// RUN: %clang_cc1 -fexperimental-strict-floating-point  %s -DSET -include-pch %t -emit-llvm -o - | FileCheck --check-prefix=CHECK-EBSTRICT %s
// RUN: %clang_cc1 -fexperimental-strict-floating-point  %s -ffp-exception-behavior=maytrap -DSET -emit-pch -o %t
// RUN: %clang_cc1 -fexperimental-strict-floating-point  %s -DSET -include-pch %t -emit-llvm -o - | FileCheck --check-prefix=CHECK-EBSTRICT %s
// RUN: %clang_cc1 -fexperimental-strict-floating-point  %s -ffp-contract=fast -DSET -emit-pch -o %t
// RUN: %clang_cc1 -fexperimental-strict-floating-point  %s -DSET -include-pch %t -emit-llvm -o - | FileCheck --check-prefix=CHECK-EBSTRICT %s
// RUN: %clang_cc1 -fexperimental-strict-floating-point  %s -DSET -emit-pch -o %t
// RUN: %clang_cc1 -fexperimental-strict-floating-point  %s -ffp-contract=on -DSET -include-pch %t -emit-llvm -o - | FileCheck --check-prefix=CHECK-CONTRACT %s
// RUN: %clang_cc1 -fexperimental-strict-floating-point  %s -DPUSH -emit-pch -o %t
// RUN: %clang_cc1 -fexperimental-strict-floating-point  %s -DPUSH -verify -include-pch %t
// RUN: %clang_cc1 -fexperimental-strict-floating-point  %s -DPUSH_POP -emit-pch -o %t
// RUN: %clang_cc1 -fexperimental-strict-floating-point  %s -DPUSH_POP -verify -include-pch %t

#ifndef HEADER
#define HEADER

#ifdef SET
#pragma float_control(except, on)
#endif

#ifdef PUSH
#pragma float_control(precise, on)
#pragma float_control(push)
#pragma float_control(precise, off)
#endif

#ifdef PUSH_POP
#pragma float_control(precise, on, push)
#pragma float_control(push)
#pragma float_control(pop)
#endif
#else

#ifdef SET
float fun(float a, float b) {
  // CHECK-LABEL: define float @fun{{.*}}
  //CHECK-EBSTRICT: llvm.experimental.constrained.fmul{{.*}}tonearest{{.*}}strict
  //CHECK-EBSTRICT: llvm.experimental.constrained.fadd{{.*}}tonearest{{.*}}strict
  //CHECK-CONTRACT: llvm.experimental.constrained.fmuladd{{.*}}tonearest{{.*}}strict
  return a * b + 2;
}
#pragma float_control(pop) // expected-warning {{#pragma float_control(pop, ...) failed: stack empty}}
#pragma float_control(pop) // expected-warning {{#pragma float_control(pop, ...) failed: stack empty}}
#endif

#ifdef PUSH
#pragma float_control(pop)
#pragma float_control(pop) // expected-warning {{#pragma float_control(pop, ...) failed: stack empty}}
#endif

#ifdef PUSH_POP
#pragma float_control(pop)
#pragma float_control(pop) // expected-warning {{#pragma float_control(pop, ...) failed: stack empty}}
#endif

#endif //ifndef HEADER
