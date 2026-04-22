// RUN: %clangxx_asan -O0 -mllvm -asan-instrument-assume-dereferenceable=1 %s -o %t && %run %t

#include <stdlib.h>

void test_pass_1() {
  char *p = (char *)malloc(20);
  __builtin_assume_dereferenceable(p, 10);
  __builtin_assume_dereferenceable(p, 20);
  free(p);
}

void test_pass_2() {
  char *p = (char *)malloc(10);
  __builtin_assume_dereferenceable(p, 0);
  free(p);
}

void test_stack_pass_1() {
  char p[20];
  __builtin_assume_dereferenceable(p, 10);
  __builtin_assume_dereferenceable(p, 20);
}

void test_stack_pass_2() {
  char p[10];
  __builtin_assume_dereferenceable(p, 0);
}

int main() {
  test_pass_1();
  test_pass_2();
  test_stack_pass_1();
  test_stack_pass_2();
  return 0;
}
