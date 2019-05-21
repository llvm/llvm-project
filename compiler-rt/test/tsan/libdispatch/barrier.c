// RUN: %clang_tsan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s --implicit-check-not='ThreadSanitizer'

#include "dispatch/dispatch.h"

#include "../test.h"

long global;

int main() {
  fprintf(stderr, "Hello world.\n");
  dispatch_semaphore_t done = dispatch_semaphore_create(0);
  barrier_init(&barrier, 2);

  dispatch_queue_t q = dispatch_queue_create("my.queue", DISPATCH_QUEUE_CONCURRENT);
  dispatch_queue_t bgq = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);

  dispatch_async(bgq, ^{
    dispatch_sync(q, ^{
      global = 42;
    });
    barrier_wait(&barrier);
  });

  dispatch_async(bgq, ^{
    barrier_wait(&barrier);
    dispatch_barrier_sync(q, ^{
      global = 43;
    });

    dispatch_async(bgq, ^{
      barrier_wait(&barrier);
      global = 44;
    });

    barrier_wait(&barrier);

    dispatch_semaphore_signal(done);
  });

  dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
  fprintf(stderr, "Done.\n");
}

// CHECK: Hello world.
// CHECK: Done.
