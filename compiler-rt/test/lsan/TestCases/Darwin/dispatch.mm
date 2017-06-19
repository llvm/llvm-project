// Test for threads spawned with wqthread_start
// RUN: LSAN_BASE="report_objects=1:use_stacks=0:use_registers=0"
// RUN: %clangxx_lsan %s -DDISPATCH_ASYNC -o %t-async -framework Foundation
// RUN: %clangxx_lsan %s -DDISPATCH_SYNC -o %t-sync -framework Foundation
// RUN: %env_lsan_opts=$LSAN_BASE not %run %t-async 2>&1 | FileCheck %s
// RUN: %env_lsan_opts=$LSAN_BASE not %run %t-sync 2>&1 | FileCheck %s

#include <dispatch/dispatch.h>
#include <pthread.h>
#include <stdlib.h>

#include "sanitizer_common/print_address.h"

bool done = false;

void worker_do_leak(int size) {
  void *p = malloc(size);
  print_address("Test alloc: ", 1, p);
  done = true;
}

#if DISPATCH_ASYNC
// Tests for the Grand Central Dispatch. See
// http://developer.apple.com/library/mac/#documentation/Performance/Reference/GCD_libdispatch_Ref/Reference/reference.html
// for the reference.
void TestGCDDispatch() {
  dispatch_queue_t queue = dispatch_get_global_queue(0, 0);
  dispatch_block_t block = ^{
    worker_do_leak(1337);
  };
  // dispatch_async() runs the task on a worker thread that does not go through
  // pthread_create(). We need to verify that LeakSanitizer notices that the
  // thread has started.
  dispatch_async(queue, block);
  while (!done)
    pthread_yield_np();
}
#elif DISPATCH_SYNC
void TestGCDDispatch() {
  dispatch_queue_t queue = dispatch_get_global_queue(2, 0);
  dispatch_block_t block = ^{
    worker_do_leak(1337);
  };
  // dispatch_sync() runs the task on a worker thread that does not go through
  // pthread_create(). We need to verify that LeakSanitizer notices that the
  // thread has started.
  dispatch_sync(queue, block);
}
#endif

int main() {
  TestGCDDispatch();
  return 0;
}

// CHECK: Test alloc: [[addr:0x[0-9,a-f]+]]
// CHECK: LeakSanitizer: detected memory leaks
// CHECK: [[addr]] (1337 bytes)
// CHECK: SUMMARY: {{(Leak|Address)}}Sanitizer:
