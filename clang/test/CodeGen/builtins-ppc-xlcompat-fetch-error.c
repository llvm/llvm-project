// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -target-cpu pwr8 \
// RUN:  -verify %s

void test_builtin_ppc_fetch_and_add2() {
  volatile int a = 0;
  unsigned int b = 0;

  __fetch_and_add(&a, b); // expected-warning {{passing 'volatile int *' to parameter of type 'volatile unsigned int *' converts between pointers to integer types with different sign}}
}

void test_builtin_ppc_fetch_and_addlp() {
  volatile long a = 0;
  unsigned long b = 0;

  __fetch_and_addlp(&a, b); // expected-warning {{passing 'volatile long *' to parameter of type 'volatile unsigned long *' converts between pointers to integer types with different sign}}
}
