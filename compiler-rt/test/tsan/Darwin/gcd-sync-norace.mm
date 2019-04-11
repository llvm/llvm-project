// RUN: %clang_tsan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

#include "dispatch/dispatch.h"

#include <stdio.h>

long global;

static const long nIter = 1000;

int main() {
  fprintf(stderr, "Hello world.\n");
  dispatch_semaphore_t done = dispatch_semaphore_create(0);

  dispatch_queue_t serial_q = dispatch_queue_create("my.queue", DISPATCH_QUEUE_SERIAL);

  global = 42;
  for (int i = 0; i < nIter; i++) {
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
      dispatch_sync(serial_q, ^{
        global = i;

        if (i == nIter - 1) {
          dispatch_semaphore_signal(done);
        }
      });
    });
  }

  dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
  fprintf(stderr, "Done.\n");
}

// CHECK: Hello world.
// CHECK: Done.
// CHECK-NOT: WARNING: ThreadSanitizer
