// RUN: %clang_tsan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s --implicit-check-not='ThreadSanitizer'

#include <dispatch/dispatch.h>

#include <stdio.h>

long global;

int main(int argc, const char *argv[]) {
  fprintf(stderr, "Hello world.\n");

  dispatch_queue_t q = dispatch_queue_create("my.queue", DISPATCH_QUEUE_SERIAL);
  dispatch_source_t timer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, q);
  long long interval_ms = 10;
  dispatch_source_set_timer(timer, dispatch_time(DISPATCH_TIME_NOW, 0), interval_ms * NSEC_PER_MSEC, 0);

  dispatch_semaphore_t sem = dispatch_semaphore_create(0);
  dispatch_source_set_event_handler(timer, ^{
    fprintf(stderr, "timer\n");
    global++;

    if (global > 50) {
      dispatch_semaphore_signal(sem);
      dispatch_suspend(timer);
    }
  });
  dispatch_resume(timer);

  dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);

  fprintf(stderr, "Done.\n");
}

// CHECK: Hello world.
// CHECK: timer
// CHECK: Done.
// CHECK-NOT: timer
