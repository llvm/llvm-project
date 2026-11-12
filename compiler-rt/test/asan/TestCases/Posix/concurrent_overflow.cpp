// RUN: %clangxx_asan -O0 -w %s -o %t && not %run %t 2>&1 | FileCheck %s

// Checks that concurrent reports will not trigger false "nested bug" reports.
// Regression test for https://github.com/google/sanitizers/issues/858

#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

static void *start_routine(void *arg) {
  volatile int *counter = (volatile int *)arg;
  char buf[8];
  __atomic_sub_fetch(counter, 1, __ATOMIC_SEQ_CST);
  while (*counter)
    ;
  buf[0] = buf[9];
  return 0;
}

int main(void) {
  const int n_threads = 8;
  int i, counter = n_threads;
  pthread_t thread[n_threads];

  for (i = 0; i < n_threads; ++i)
    pthread_create(&thread[i], NULL, &start_routine, (void *)&counter);
  for (i = 0; i < n_threads; ++i)
    pthread_join(thread[i], NULL);
  return 0;
}

// CHECK-NOT: nested bug
// CHECK: ERROR: AddressSanitizer: stack-buffer-overflow on address
// CHECK: SUMMARY: AddressSanitizer: stack-buffer-overflow
