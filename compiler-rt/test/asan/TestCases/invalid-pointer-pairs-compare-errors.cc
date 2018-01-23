// RUN: %clangxx_asan -O0 %s -o %t -mllvm -asan-detect-invalid-pointer-pair

// RUN: %env_asan_opts=detect_invalid_pointer_pairs=1:halt_on_error=0 %run %t 2>&1 | FileCheck %s

#include <assert.h>
#include <stdlib.h>

int foo(char *p, char *q) {
  return p > q;
}

char global1[100] = {}, global2[100] = {};
char __attribute__((used)) smallest_global[5] = {};
char small_global[7] = {};
char __attribute__((used)) little_global[10] = {};
char __attribute__((used)) medium_global[4000] = {};
char large_global[5000] = {};
char __attribute__((used)) largest_global[6000] = {};

int main() {
  // Heap allocated memory.
  char *heap1 = (char *)malloc(42);
  char *heap2 = (char *)malloc(42);

  // CHECK: ERROR: AddressSanitizer: invalid-pointer-pair
  // CHECK: #{{[0-9]+ .*}} in main {{.*}}invalid-pointer-pairs-compare-errors.cc:[[@LINE+1]]
  foo(heap1, heap2);
  free(heap1);
  free(heap2);

  heap1 = (char *)malloc(1024);
  // CHECK: ERROR: AddressSanitizer: invalid-pointer-pair
  // CHECK: #{{[0-9]+ .*}} in main {{.*}}invalid-pointer-pairs-compare-errors.cc:[[@LINE+1]]
  foo(heap1, heap1 + 1025);
  // CHECK: ERROR: AddressSanitizer: invalid-pointer-pair
  // CHECK: #{{[0-9]+ .*}} in main {{.*}}invalid-pointer-pairs-compare-errors.cc:[[@LINE+1]]
  foo(heap1 + 1024, heap1 + 1025);
  free(heap1);

  heap1 = (char *)malloc(4096);
  // CHECK: ERROR: AddressSanitizer: invalid-pointer-pair
  // CHECK: #{{[0-9]+ .*}} in main {{.*}}invalid-pointer-pairs-compare-errors.cc:[[@LINE+1]]
  foo(heap1, heap1 + 4097);
  // CHECK: ERROR: AddressSanitizer: invalid-pointer-pair
  // CHECK: #{{[0-9]+ .*}} in main {{.*}}invalid-pointer-pairs-compare-errors.cc:[[@LINE+1]]
  foo(heap1, 0);

  // Global variables.
  // CHECK: ERROR: AddressSanitizer: invalid-pointer-pair
  // CHECK: #{{[0-9]+ .*}} in main {{.*}}invalid-pointer-pairs-compare-errors.cc:[[@LINE+1]]
  foo(&global1[0], &global2[10]);

  char *p = &small_global[0];
  foo(p, p); // OK
  foo(p, p + 7); // OK
  // CHECK: ERROR: AddressSanitizer: invalid-pointer-pair
  // CHECK: #{{[0-9]+ .*}} in main {{.*}}invalid-pointer-pairs-compare-errors.cc:[[@LINE+1]]
  foo(p, p + 8);
  // CHECK: ERROR: AddressSanitizer: invalid-pointer-pair
  // CHECK: #{{[0-9]+ .*}} in main {{.*}}invalid-pointer-pairs-compare-errors.cc:[[@LINE+1]]
  foo(p - 1, p);
  // CHECK: ERROR: AddressSanitizer: invalid-pointer-pair
  // CHECK: #{{[0-9]+ .*}} in main {{.*}}invalid-pointer-pairs-compare-errors.cc:[[@LINE+1]]
  foo(p, p - 1);
  // CHECK: ERROR: AddressSanitizer: invalid-pointer-pair
  // CHECK: #{{[0-9]+ .*}} in main {{.*}}invalid-pointer-pairs-compare-errors.cc:[[@LINE+1]]
  foo(p - 1, p + 8);

  p = &large_global[0];
  // CHECK: ERROR: AddressSanitizer: invalid-pointer-pair
  // CHECK: #{{[0-9]+ .*}} in main {{.*}}invalid-pointer-pairs-compare-errors.cc:[[@LINE+1]]
  foo(p - 1, p);
  // CHECK: ERROR: AddressSanitizer: invalid-pointer-pair
  // CHECK: #{{[0-9]+ .*}} in main {{.*}}invalid-pointer-pairs-compare-errors.cc:[[@LINE+1]]
  foo(p, p - 1);
  // CHECK: ERROR: AddressSanitizer: invalid-pointer-pair
  // CHECK: #{{[0-9]+ .*}} in main {{.*}}invalid-pointer-pairs-compare-errors.cc:[[@LINE+1]]
  foo(p, &global1[0]);
  // CHECK: ERROR: AddressSanitizer: invalid-pointer-pair
  // CHECK: #{{[0-9]+ .*}} in main {{.*}}invalid-pointer-pairs-compare-errors.cc:[[@LINE+1]]
  foo(p, &small_global[0]);
  // CHECK: ERROR: AddressSanitizer: invalid-pointer-pair
  // CHECK: #{{[0-9]+ .*}} in main {{.*}}invalid-pointer-pairs-compare-errors.cc:[[@LINE+1]]
  foo(p, 0);

  // Stack variables.
  char stack1, stack2;
  // CHECK: ERROR: AddressSanitizer: invalid-pointer-pair
  // CHECK: #{{[0-9]+ .*}} in main {{.*}}invalid-pointer-pairs-compare-errors.cc:[[@LINE+1]]
  foo(&stack1, &stack2);

  // Mixtures.
  // CHECK: ERROR: AddressSanitizer: invalid-pointer-pair
  // CHECK: #{{[0-9]+ .*}} in main {{.*}}invalid-pointer-pairs-compare-errors.cc:[[@LINE+1]]
  foo(heap1, &stack1);
  // CHECK: ERROR: AddressSanitizer: invalid-pointer-pair
  foo(heap1, &global1[0]);
  // CHECK: ERROR: AddressSanitizer: invalid-pointer-pair
  foo(&stack1, &global1[0]);
  // CHECK: ERROR: AddressSanitizer: invalid-pointer-pair
  // CHECK: #{{[0-9]+ .*}} in main {{.*}}invalid-pointer-pairs-compare-errors.cc:[[@LINE+1]]
  foo(&stack1, 0);

  free(heap1);

  return 0;
}
