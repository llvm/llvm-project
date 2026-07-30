// RUN: %clang_cc1 -fopenmp -fsyntax-only -triple aarch64 -target-feature +sve -verify %s
// expected-no-diagnostics

__SVBool_t foo(int);

void test() {
#pragma omp parallel
  {
    __SVBool_t pg = foo(1);
  }
}
