// REQUIRES: x86-registered-target

// With the pragma in place, generic targets leave FMA enabled unless the
// switch '-ffp-contract=fast-honor-pragmas' is used to disable it; whereas
// for PlayStation, the pragma is always honored, so FMA is disabled even in
// plain 'fast' mode:
// RUN: %clang_cc1 -S -triple x86_64-unknown-unknown -target-feature +fma \
// RUN:   -O2 -ffp-contract=fast -o - %s | \
// RUN:   FileCheck --check-prefix=CHECK-YES-FMA %s
// RUN: %clang_cc1 -S -triple x86_64-unknown-unknown -target-feature +fma \
// RUN:   -O2 -ffp-contract=fast-honor-pragmas -o - %s | \
// RUN:   FileCheck --check-prefix=CHECK-NO-FMA %s
// RUN: %clang_cc1 -S -triple x86_64-unknown-unknown -target-feature +fma \
// RUN:   -O2 -ffp-contract=fast -ffp-contract=fast-honor-pragmas -o - %s | \
// RUN:   FileCheck --check-prefix=CHECK-NO-FMA %s
// RUN: %clang_cc1 -S -triple x86_64-sie-ps5 -target-feature +fma \
// RUN:   -O2 -ffp-contract=fast -o - %s | \
// RUN:   FileCheck --check-prefix=CHECK-NO-FMA %s
// RUN: %clang_cc1 -S -triple x86_64-sie-ps5 -target-feature +fma \
// RUN:   -O2 -ffp-contract=fast-honor-pragmas -o - %s | \
// RUN:   FileCheck --check-prefix=CHECK-NO-FMA %s
//
// With the pragma suppressed, FMA happens in 'fast' or 'fast-honor-pragmas'
// modes (for generic targets and for PlayStation):
// RUN: %clang_cc1 -S -DSUPPRESS_PRAGMA \
// RUN:   -triple x86_64-unknown-unknown -target-feature +fma \
// RUN:   -O2 -ffp-contract=fast -o - %s | \
// RUN:   FileCheck --check-prefix=CHECK-YES-FMA %s
// RUN: %clang_cc1 -S -DSUPPRESS_PRAGMA \
// RUN:   -triple x86_64-unknown-unknown -target-feature +fma \
// RUN:   -O2 -ffp-contract=fast-honor-pragmas -o - %s | \
// RUN:   FileCheck --check-prefix=CHECK-YES-FMA %s
// RUN: %clang_cc1 -S -DSUPPRESS_PRAGMA \
// RUN:   -triple x86_64-sie-ps5 -target-feature +fma \
// RUN:   -O2 -ffp-contract=fast -o - %s | \
// RUN:   FileCheck --check-prefix=CHECK-YES-FMA %s
// RUN: %clang_cc1 -S -DSUPPRESS_PRAGMA \
// RUN:   -triple x86_64-sie-ps5 -target-feature +fma \
// RUN:   -O2 -ffp-contract=fast-honor-pragmas -o - %s | \
// RUN:   FileCheck --check-prefix=CHECK-YES-FMA %s
//
float compute(float a, float b, float c) {
#if !defined(SUPPRESS_PRAGMA)
#pragma clang fp contract (off)
#endif
  float product = a * b;
  return product + c;
}

// CHECK-NO-FMA: vmulss
// CHECK-NO-FMA-NEXT: vaddss

// CHECK-YES-FMA: vfmadd213ss
