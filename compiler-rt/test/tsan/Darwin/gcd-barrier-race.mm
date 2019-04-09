// RUN: %clang_tsan %s -o %t
// RUN: %deflake %run %t 2>&1 | FileCheck %s

#include "dispatch/dispatch.h"

#include "../test.h"

long global;

int main() {
  fprintf(stderr, "Hello world.\n");
  print_address("addr=", 1, &global);
  dispatch_semaphore_t done = dispatch_semaphore_create(0);
  barrier_init(&barrier, 2);

  dispatch_queue_t q = dispatch_queue_create("my.queue", DISPATCH_QUEUE_CONCURRENT);
  dispatch_queue_t bgq = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);

  dispatch_barrier_sync(q, ^{
    global = 42;
  });

  dispatch_async(bgq, ^{
    dispatch_sync(q, ^{
      global = 43;
      barrier_wait(&barrier);
    });
  });

  dispatch_async(bgq, ^{
    dispatch_sync(q, ^{
      barrier_wait(&barrier);
      global = 44;

      dispatch_semaphore_signal(done);
    });
  });

  dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
  fprintf(stderr, "Done.\n");
}

// CHECK: Hello world.
// CHECK: addr=[[ADDR:0x[0-9,a-f]+]]
// CHECK: WARNING: ThreadSanitizer: data race
// CHECK: Location is global 'global' {{(of size 8 )?}}at [[ADDR]] (gcd-barrier-race.mm.tmp+0x{{[0-9,a-f]+}})
// CHECK: Done.
