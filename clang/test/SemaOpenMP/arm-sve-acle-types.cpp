// RUN: %clang_cc1 -fopenmp -fsyntax-only -triple aarch64 -target-feature +sve -verify %s

__SVBool_t foo(int);

void test() {
#pragma omp parallel
  {
    __SVBool_t pg = foo(1);
  }
}

// A sizeless type has no size to shape.
void shaping(__SVInt32_t *svp, int n) {
#pragma omp target update to(([n])svp) // expected-error {{expected expression with a pointer to a complete type as a base of an array shaping operation}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update from((([n])svp)[0:n]) // expected-error {{expected expression with a pointer to a complete type as a base of an array shaping operation}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
}
