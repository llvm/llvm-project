// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -target-cpu pwr8 \
// RUN:   -verify %s

void test_builtin_ppc_compare_and_swap() {
  volatile int a = 0;
  long b = 0, c = 0;

  __compare_and_swap(&a, &b, c); // expected-warning {{incompatible pointer types passing 'long *' to parameter of type 'int *'}}

}

void test_builtin_ppc_compare_and_swaplp() {
  volatile long a = 0;
  int b = 0, c = 0;

  __compare_and_swaplp(&a, &b, c);// expected-warning {{incompatible pointer types passing 'int *' to parameter of type 'long *'}}

}
