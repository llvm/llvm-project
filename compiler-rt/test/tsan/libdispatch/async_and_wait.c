// RUN: %clang_tsan %s -o %t -undefined dynamic_lookup
// RUN: %run %t 2>&1 | FileCheck %s --implicit-check-not='ThreadSanitizer'

#include "dispatch/dispatch.h"

#include <stdio.h>

// Allow compilation with pre-macOS 10.14 (and aligned) SDKs
API_AVAILABLE(macos(10.14), ios(12.0), tvos(12.0), watchos(5.0))
DISPATCH_EXPORT DISPATCH_NONNULL_ALL DISPATCH_NOTHROW
void dispatch_async_and_wait(dispatch_queue_t queue,
           DISPATCH_NOESCAPE dispatch_block_t block);

long global;

int main() {
  // Guard execution on pre-macOS 10.14 (and aligned) platforms
  if (dispatch_async_and_wait == NULL) {
    fprintf(stderr, "Done.\n");
    return 0;
  }

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
  return 0;
}

// CHECK: Done.
