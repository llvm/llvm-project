// RUN: %clang_tsan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s --implicit-check-not='ThreadSanitizer'

#include "dispatch/dispatch.h"

#include <stdio.h>

long global;

int main() {
  dispatch_queue_t q = dispatch_queue_create("my.queue", DISPATCH_QUEUE_SERIAL);
  dispatch_semaphore_t s = dispatch_semaphore_create(0);

  // Force queue to context switch onto separate thread.
  dispatch_async(q, ^{
    dispatch_semaphore_wait(s, DISPATCH_TIME_FOREVER);
  });
  dispatch_semaphore_signal(s);

  global++;
  dispatch_async_and_wait(q, ^{
    // The queue continues to execute on separate thread.  This would cause a
    // race if we had used `dispatch_async()` without the `_and_wait` part.
    global++;
  });
  global++;

  fprintf(stderr, "Done.\n");
}

// CHECK: Done.
