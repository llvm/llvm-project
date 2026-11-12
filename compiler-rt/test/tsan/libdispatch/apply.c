// RUN: %clang_tsan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s --implicit-check-not='ThreadSanitizer'

#include <dispatch/dispatch.h>

#include "../test.h"

const size_t size = 2;
long global;
long array[size];

void callback(void *context, size_t i) {
  long n = global;
  array[i] = n + i;
  barrier_wait(&barrier);
}

int main(int argc, const char *argv[]) {
  fprintf(stderr, "start\n");

  // Warm up GCD (workaround for macOS Sierra where dispatch_apply might run single-threaded).
  dispatch_sync(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{ });

  dispatch_queue_t q = dispatch_queue_create("my.queue", DISPATCH_QUEUE_CONCURRENT);

  global = 42;

  barrier_init(&barrier, size);
  dispatch_apply(size, q, ^(size_t i) {
    long n = global;
    array[i] = n + i;
    barrier_wait(&barrier);
  });

  for (size_t i = 0; i < size; i++) {
    fprintf(stderr, "array[%ld] = %ld\n", i, array[i]);
  }

  global = 142;

  barrier_init(&barrier, size);
  dispatch_apply_f(size, q, NULL, &callback);

  for (size_t i = 0; i < size; i++) {
    fprintf(stderr, "array[%ld] = %ld\n", i, array[i]);
  }

  fprintf(stderr, "done\n");
  return 0;
}

// CHECK: start
// CHECK: array[0] = 42
// CHECK: array[1] = 43
// CHECK: array[0] = 142
// CHECK: array[1] = 143
// CHECK: done
