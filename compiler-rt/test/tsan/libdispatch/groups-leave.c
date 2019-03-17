// RUN: %clang_tsan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

#include <dispatch/dispatch.h>

#include "../test.h"

dispatch_semaphore_t sem;

long global;
long global2;

void callback(void *context) {
  global2 = 48;
  barrier_wait(&barrier);

  dispatch_semaphore_signal(sem);
}

int main() {
  fprintf(stderr, "Hello world.\n");
  barrier_init(&barrier, 2);

  dispatch_queue_t q = dispatch_queue_create("my.queue", DISPATCH_QUEUE_CONCURRENT);
  dispatch_group_t g = dispatch_group_create();
  sem = dispatch_semaphore_create(0);

  dispatch_group_enter(g);
  dispatch_async(q, ^{
    global = 47;
    dispatch_group_leave(g);
    barrier_wait(&barrier);
  });
  dispatch_group_notify(g, q, ^{
    global = 48;
    barrier_wait(&barrier);

    dispatch_semaphore_signal(sem);
  });
  dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);

  dispatch_group_enter(g);
  dispatch_async(q, ^{
    global2 = 47;
    dispatch_group_leave(g);
    barrier_wait(&barrier);
  });
  dispatch_group_notify_f(g, q, NULL, &callback);
  dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);

  fprintf(stderr, "Done.\n");
}

// CHECK: Hello world.
// CHECK-NOT: WARNING: ThreadSanitizer
// CHECK: Done.
