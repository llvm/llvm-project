// RUN: %clang_tsan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s --implicit-check-not='ThreadSanitizer'

#include "dispatch/dispatch.h"

#include <stdio.h>

long global;

int main() {
  fprintf(stderr, "Hello world.\n");
  dispatch_semaphore_t done = dispatch_semaphore_create(0);

  dispatch_queue_t q1 = dispatch_queue_create("my.queue1", DISPATCH_QUEUE_CONCURRENT);
  dispatch_queue_t q2 = dispatch_queue_create("my.queue2", DISPATCH_QUEUE_SERIAL);

  global = 42;
  for (int i = 0; i < 10; i++) {
    dispatch_async(q1, ^{
      for (int i = 0; i < 100; i++) {
        dispatch_sync(q2, ^{
          global++;
        });
      }
    });
  }

  dispatch_barrier_async(q1, ^{
    dispatch_semaphore_signal(done);
  });

  dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
  fprintf(stderr, "Done.\n");
}

// CHECK: Hello world.
// CHECK: Done.
