// RUN: %clang_tsan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s --implicit-check-not='ThreadSanitizer'

#include "dispatch/dispatch.h"

#include <stdio.h>
#include <assert.h>

int main() {
  fprintf(stderr, "start\n");
  dispatch_semaphore_t done = dispatch_semaphore_create(0);

  dispatch_queue_t background_q = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
  dispatch_queue_t serial_q = dispatch_queue_create("my.queue", DISPATCH_QUEUE_SERIAL);
  assert(background_q != serial_q);

  dispatch_async(background_q, ^{
    __block long block_var = 0;

    dispatch_sync(serial_q, ^{
      block_var = 42;
    });

    fprintf(stderr, "block_var = %ld\n", block_var);

    dispatch_semaphore_signal(done);
  });

  dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
  fprintf(stderr, "done\n");
}

// CHECK: start
// CHECK: block_var = 42
// CHECK: done
