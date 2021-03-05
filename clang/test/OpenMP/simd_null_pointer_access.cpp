// RUN: %clang_cc1 -fopenmp-simd -fsycl -fsycl-is-device -triple spir64 -verify -fsyntax-only %s

// Test that in the presence of SYCL options, that null function
// declarations are accounted for when checking to emit diagnostics.

// expected-no-diagnostics

__thread void *x;
