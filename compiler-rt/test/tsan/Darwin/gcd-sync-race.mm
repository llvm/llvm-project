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

  dispatch_queue_t q1 = dispatch_queue_create("my.queue1", DISPATCH_QUEUE_CONCURRENT);
  dispatch_queue_t q2 = dispatch_queue_create("my.queue2", DISPATCH_QUEUE_CONCURRENT);

  global = 42;
  dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
    dispatch_sync(q1, ^{
      global = 43;
      barrier_wait(&barrier);
    });
  });
  dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
    dispatch_sync(q2, ^{
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
// CHECK: Location is global 'global' {{(of size 8 )?}}at [[ADDR]] (gcd-sync-race.mm.tmp+0x{{[0-9,a-f]+}})
// CHECK: Done.
