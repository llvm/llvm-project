// RUN: not %clang_cc1 -fexperimental-strict-floating-point \
// RUN: -triple x86_64-linux-gnu  \
// RUN: -verify %s 2>&1 | FileCheck %s --check-prefixes=CHECK-PRGM

// RUN: not %clang_cc1 -fexperimental-strict-floating-point \
// RUN: -triple x86_64-linux-gnu  -freciprocal-math \
// RUN: -verify %s 2>&1 | FileCheck %s --check-prefixes=CHECK-RECPR,CHECK-PRGM

// RUN: not %clang_cc1 -fexperimental-strict-floating-point \
// RUN: -triple x86_64-linux-gnu  -mreassociate \
// RUN: -verify %s 2>&1 | FileCheck %s --check-prefixes=CHECK-ASSOC,CHECK-PRGM

// RUN: not %clang_cc1 -fexperimental-strict-floating-point \
// RUN: -triple x86_64-linux-gnu  -fapprox-func \
// RUN: -verify %s 2>&1 | FileCheck %s --check-prefixes=CHECK-FUNC,CHECK-PRGM

// RUN: not %clang_cc1 -fexperimental-strict-floating-point \
// RUN: -triple x86_64-linux-gnu -freciprocal-math -mreassociate -verify \
// RUN: %s 2>&1 | FileCheck %s --check-prefixes=CHECK-ASSOC,CHECK-RECPR,CHECK-PRGM

// RUN: not %clang_cc1 -fexperimental-strict-floating-point \
// RUN: -triple x86_64-linux-gnu -freciprocal-math -mreassociate -fapprox-func \
// RUN: -verify %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHECK-FUNC,CHECK-ASSOC,CHECK-RECPR,CHECK-PRGM

// RUN: not %clang_cc1 -fexperimental-strict-floating-point \
// RUN: -triple x86_64-linux-gnu -ffp-eval-method=source \
// RUN: -verify %s 2>&1 | FileCheck %s --check-prefixes=CHECK-FFP-OPT,CHECK-PRGM

// expected-no-diagnostics

float f1(float a, float b, float c) {
  a = b + c;
  return a * b + c;
}

float f2(float a, float b, float c) {
  // CHECK-FFP-OPT: option 'ffp-eval-method' cannot be used with '#pragma clang fp reassociate'
#pragma clang fp reassociate(on)
  return (a + b) + c;
}

float f3(float a, float b, float c) {
#pragma clang fp reassociate(off)
  return (a - b) - c;
}

float f4(float a, float b, float c) {
#pragma clang fp eval_method(double)
  // CHECK-FUNC: '#pragma clang fp eval_method' cannot be used with option 'fapprox-func'
  // CHECK-ASSOC: '#pragma clang fp eval_method' cannot be used with option 'mreassociate'
  // CHECK-RECPR: '#pragma clang fp eval_method' cannot be used with option 'freciprocal'
  // CHECK-PRGM: '#pragma clang fp eval_method' cannot be used with '#pragma clang fp reassociate'
#pragma clang fp reassociate(on)
  return (a * c) - (b * c);
}
