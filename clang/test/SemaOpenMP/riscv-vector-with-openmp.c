// RUN: %clang_cc1 -triple riscv64 -target-feature +v -fsyntax-only \
// RUN: -verify -fopenmp %s
// REQUIRES: riscv-registered-target

// expected-no-diagnostics

void foo() {
  #pragma omp parallel
  {
    __rvv_int32m1_t i32m1;
  }
}
