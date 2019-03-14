// RUN: %clang_tsan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

// TODO(yln): Deadlocks while gcd-apply.mm does not. What's the difference
//            between C and Obj-C compiler?
// REQUIRES: disable

#include <dispatch/dispatch.h>

#include "../test.h"

long global;
long array[2];

void callback(void *context, size_t i) {
  long n = global;
  array[i] = n + i;
  barrier_wait(&barrier);
}

int main(int argc, const char *argv[]) {
  barrier_init(&barrier, 2);
  fprintf(stderr, "start\n");

  // Warm up GCD (workaround for macOS Sierra where dispatch_apply might run single-threaded).
  dispatch_sync(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{ });

  dispatch_queue_t q = dispatch_queue_create("my.queue", DISPATCH_QUEUE_CONCURRENT);

  global = 42;

  dispatch_apply(100, q, ^(size_t i) {
    long n = global;
    array[i] = n + i;
    barrier_wait(&barrier);
  });

  for (int i = 0; i < 100; i++) {
    fprintf(stderr, "array[%d] = %ld\n", i, array[i]);
  }

  global = 43;

  dispatch_apply_f(100, q, NULL, &callback);

  fprintf(stderr, "done\n");
  return 0;
}

// CHECK: start
// CHECK: done
// CHECK-NOT: WARNING: ThreadSanitizer
