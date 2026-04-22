// RUN: %clangxx_asan -O0 -mllvm -asan-instrument-assume-dereferenceable=1 -fsanitize-recover=address %s -o %t && %env_asan_opts=halt_on_error=0 %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>

void test_malloc_fully_oob() {
  char *p = (char *)malloc(10);
  fprintf(stderr, "test_malloc_fully_oob\n");
  // CHECK: test_malloc_fully_oob
  // CHECK: ERROR: AddressSanitizer: dereferenceable-assumption-violation on address [[PTR1:0x[0-9a-fA-F]+]]
  // CHECK: ASSUME of size 20 at [[PTR1]] thread T0
  __builtin_assume_dereferenceable(p, 20);
  free(p);
}

void test_malloc_partial_right() {
  char *p = (char *)malloc(10);
  fprintf(stderr, "test_malloc_partial_right\n");
  // CHECK: test_malloc_partial_right
  // CHECK: ERROR: AddressSanitizer: dereferenceable-assumption-violation on address [[PTR2:0x[0-9a-fA-F]+]]
  // CHECK: ASSUME of size 10 at [[PTR2]] thread T0
  __builtin_assume_dereferenceable(p + 5, 10);
  free(p);
}

void test_malloc_partial_left() {
  char *p = (char *)malloc(10);
  fprintf(stderr, "test_malloc_partial_left\n");
  // CHECK: test_malloc_partial_left
  // CHECK: ERROR: AddressSanitizer: dereferenceable-assumption-violation on address [[PTR3:0x[0-9a-fA-F]+]]
  // CHECK: ASSUME of size 10 at [[PTR3]] thread T0
  __builtin_assume_dereferenceable(p - 5, 10);
  free(p);
}

void test_stack_fully_oob(int i) {
  char p[10];
  fprintf(stderr, "test_stack_fully_oob\n");
  // CHECK: test_stack_fully_oob
  // CHECK: ERROR: AddressSanitizer: dereferenceable-assumption-violation on address [[PTR4:0x[0-9a-fA-F]+]]
  // CHECK: ASSUME of size 100 at [[PTR4]] thread T0
  __builtin_assume_dereferenceable(p, 100);

  // This is here just to force ASan to emit instrumentation for poisoning
  // redzones around the stack. By default, ASan will not instrument stack
  // allocations that it deems "uninteresting".
  p[i] = 0;
}

void test_stack_partial_right() {
  char p[10];
  fprintf(stderr, "test_stack_partial_right\n");
  // CHECK: test_stack_partial_right
  // CHECK: ERROR: AddressSanitizer: dereferenceable-assumption-violation on address [[PTR5:0x[0-9a-fA-F]+]]
  // CHECK: ASSUME of size 10 at [[PTR5]] thread T0
  __builtin_assume_dereferenceable(p + 5, 10);
}

void test_stack_partial_left() {
  char p[10];
  fprintf(stderr, "test_stack_partial_left\n");
  // CHECK: test_stack_partial_left
  // CHECK: ERROR: AddressSanitizer: dereferenceable-assumption-violation on address [[PTR6:0x[0-9a-fA-F]+]]
  // CHECK: ASSUME of size 10 at [[PTR6]] thread T0
  __builtin_assume_dereferenceable(p - 5, 10);
}

void test_malloc_completely_poisoned() {
  char *p = (char *)malloc(10);
  free(p);
  fprintf(stderr, "test_malloc_completely_poisoned\n");
  // CHECK: test_malloc_completely_poisoned
  // CHECK: ERROR: AddressSanitizer: dereferenceable-assumption-violation on address [[PTR7:0x[0-9a-fA-F]+]]
  // CHECK: ASSUME of size 10 at [[PTR7]] thread T0
  __builtin_assume_dereferenceable(p, 10);
}

int main() {
  test_malloc_fully_oob();
  test_malloc_partial_right();
  test_malloc_partial_left();
  test_stack_fully_oob(0);
  test_stack_partial_right();
  test_stack_partial_left();
  test_malloc_completely_poisoned();
  return 0;
}
