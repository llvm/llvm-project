// Stress test recovery mode with many threads.
//
// RUN: %clangxx_asan -fsanitize-recover=address -pthread %s -o %t
//
// RUN: %env_asan_opts=halt_on_error=false:suppress_equal_pcs=false %run %t 1 10 >%t.log 2>&1
// RUN: grep 'ERROR: AddressSanitizer: use-after-poison' %t.log | count 10
// RUN: FileCheck %s <%t.log
//
// RUN: %env_asan_opts=halt_on_error=false:suppress_equal_pcs=false:exitcode=0 %run %t 10 20 >%t.log 2>&1
// RUN: grep 'ERROR: AddressSanitizer: use-after-poison' %t.log | count 200
// RUN: FileCheck %s <%t.log
//
// RUN: %env_asan_opts=halt_on_error=false:exitcode=0 %run %t 10 20 >%t.log 2>&1
// RUN: grep 'ERROR: AddressSanitizer: use-after-poison' %t.log | count 1
// RUN: FileCheck %s <%t.log

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#include <sanitizer/asan_interface.h>

size_t nthreads = 10;
size_t niter = 10;

void random_delay(unsigned *seed) {
  *seed = 1664525 * *seed + 1013904223;
  struct timespec delay = { 0, static_cast<long>((*seed % 1000) * 1000) };
  nanosleep(&delay, 0);
}

void *run(void *arg) {
  unsigned seed = (unsigned)(size_t)arg;

  volatile char tmp[2];
  __asan_poison_memory_region(&tmp, sizeof(tmp));

  for (size_t i = 0; i < niter; ++i) {
    random_delay(&seed);
    // CHECK: ERROR: AddressSanitizer: use-after-poison
    volatile int idx = 0;
    tmp[idx] = 0;
  }

  return 0;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "Syntax: %s nthreads niter\n", argv[0]);
    exit(1);
  }

  nthreads = (size_t)strtoul(argv[1], 0, 0);
  niter = (size_t)strtoul(argv[2], 0, 0);

  pthread_t *tids = new pthread_t[nthreads];

  for (size_t i = 0; i < nthreads; ++i) {
    if (0 != pthread_create(&tids[i], 0, run, (void *)i)) {
      fprintf(stderr, "Failed to create thread\n");
      exit(1);
    }
  }

  for (size_t i = 0; i < nthreads; ++i) {
    if (0 != pthread_join(tids[i], 0)) {
      fprintf(stderr, "Failed to join thread\n");
      exit(1);
    }
  }

  // CHECK: All threads terminated
  printf("All threads terminated\n");

  delete [] tids;

  return 0;
}
