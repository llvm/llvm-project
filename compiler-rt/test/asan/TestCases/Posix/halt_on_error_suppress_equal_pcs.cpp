// Test reports dedupication for recovery mode.
//
// RUN: %clang_asan -fsanitize-recover=address %s -o %t
//
// Check for reports dedupication.
// RUN: %env_asan_opts=halt_on_error=false %run %t 2>&1 | FileCheck %s
//
// Check that we die after reaching different reports number threshold.
// RUN: %env_asan_opts=halt_on_error=false not %run %t 1 >%t1.log 2>&1
// RUN: grep 'ERROR: AddressSanitizer: stack-buffer-overflow' %t1.log | count 25
//
// Check suppress_equal_pcs=true behavior is equal to default one.
// RUN: %env_asan_opts=halt_on_error=false:suppress_equal_pcs=true %run %t 2>&1 | FileCheck %s
//
// Check suppress_equal_pcs=false behavior isn't equal to default one.
// RUN: rm -f %t2.log
// RUN: %env_asan_opts=halt_on_error=false:suppress_equal_pcs=false %run %t >%t2.log 2>&1
// RUN: grep 'ERROR: AddressSanitizer: stack-buffer-overflow' %t2.log | count 30

#define ACCESS_ARRAY_FIVE_ELEMENTS(array, i)     \
  array[i] = i;                                  \
  array[i + 1] = i + 1;                          \
  array[i + 2] = i + 2;                          \
  array[i + 3] = i + 3;                          \
  array[i + 4] = i + 4;                          \

volatile int ten = 10;
unsigned kNumIterations = 10;

int main(int argc, char **argv) {
  char a[10];
  char b[10];

  if (argc == 1) {
    for (int i = 0; i < kNumIterations; ++i) {
      // CHECK: READ of size 1
      volatile int res = a[ten + i];
      // CHECK: WRITE of size 1
      a[i + ten] = res + 3;
      // CHECK: READ of size 1
      res = a[ten + i];
      // CHECK-NOT: ERROR
    }
  } else {
    for (int i = 0; i < kNumIterations; ++i) {
      ACCESS_ARRAY_FIVE_ELEMENTS(a, ten);
      ACCESS_ARRAY_FIVE_ELEMENTS(a, ten + 5);
      ACCESS_ARRAY_FIVE_ELEMENTS(a, ten + 10);
      ACCESS_ARRAY_FIVE_ELEMENTS(b, ten);
      ACCESS_ARRAY_FIVE_ELEMENTS(b, ten + 5);
      ACCESS_ARRAY_FIVE_ELEMENTS(b, ten + 10);
    }
  }
  return 0;
}

