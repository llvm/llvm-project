// RUN: %clang_tsan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s --implicit-check-not='ThreadSanitizer'

#include "dispatch/dispatch.h"

#include <stdio.h>

long global;

int main(int argc, const char *argv[]) {
  dispatch_semaphore_t done = dispatch_semaphore_create(0);
  dispatch_queue_t target_queue = dispatch_queue_create(NULL, DISPATCH_QUEUE_SERIAL);
  dispatch_queue_t q1 = dispatch_queue_create(NULL, DISPATCH_QUEUE_CONCURRENT);
  dispatch_queue_t q2 = dispatch_queue_create(NULL, DISPATCH_QUEUE_CONCURRENT);
  dispatch_set_target_queue(q1, target_queue);
  dispatch_set_target_queue(q2, target_queue);

  for (int i = 0; i < 100000; i++) {
    dispatch_async(q1, ^{
      global++;

      if (global == 200000) {
        dispatch_semaphore_signal(done);
      }
    });
    dispatch_async(q2, ^{
      global++;

      if (global == 200000) {
        dispatch_semaphore_signal(done);
      }
    });
  }

  dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
  fprintf(stderr, "Done.\n");
  return 0;
}

// CHECK: Done.
