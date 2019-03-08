// RUN: %clang_tsan %s -o %t
// RUN: %deflake %run %t 2>&1 | FileCheck %s

// REQUIRES: dispatch

#include <dispatch/dispatch.h>

#include "test.h"

long global;

int main(int argc, const char *argv[]) {
  barrier_init(&barrier, 2);
  fprintf(stderr, "start\n");

  // Warm up GCD (workaround for macOS Sierra where dispatch_apply might run single-threaded).
  dispatch_sync(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{ });

  dispatch_queue_t q = dispatch_queue_create("my.queue", DISPATCH_QUEUE_CONCURRENT);
  dispatch_apply(2, q, ^(size_t i) {
    global = i;
    barrier_wait(&barrier);
  });

  fprintf(stderr, "done\n");
  return 0;
}

// CHECK: start
// CHECK: WARNING: ThreadSanitizer: data race
// CHECK: Location is global 'global'
// CHECK: done
