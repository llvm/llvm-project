// RUN: %clang_cc1 -triple riscv64 -target-feature +v -fsyntax-only \
// RUN: -verify -fopenmp %s
// REQUIRES: riscv-registered-target

void foo() {
  #pragma omp parallel
  {
    __rvv_int32m1_t i32m1;
  }
}

// A sizeless type has no size to shape.
void shaping(__rvv_int32m1_t *vp, int n) {
#pragma omp target update to(([n])vp) // expected-error {{expected expression with a pointer to a complete type as a base of an array shaping operation}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
}
