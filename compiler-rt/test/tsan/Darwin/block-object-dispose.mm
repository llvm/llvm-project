// Test that disposing a heap-allocated __block (byref) variable synchronizes
// the non-final release(s) with the final release that runs
// __Block_byref_object_dispose_ and frees the Block_byref storage.

// RUN: %clangxx_tsan %s -o %t -framework Foundation
// RUN: %env_tsan_opts=ignore_interceptors_accesses=0:ignore_noninstrumented_modules=0 %run %t 2>&1 | FileCheck %s --implicit-check-not='ThreadSanitizer'
// RUN: %env_tsan_opts=ignore_interceptors_accesses=1:ignore_noninstrumented_modules=1 %run %t 2>&1 | FileCheck %s --implicit-check-not='ThreadSanitizer'

#import <Block.h>
#import <Foundation/Foundation.h>

#import "../test.h"

long my_global;
invisible_barrier_t barrier2;

struct MyStruct {
  bool on_heap = false;
  MyStruct() = default;
  MyStruct(const MyStruct &) : on_heap(true) {}
  void setGlobal() { my_global = 42; }
  ~MyStruct() {
    if (on_heap)
      my_global = 43;
  }
};

int main(int argc, const char *argv[]) {
  fprintf(stderr, "Hello world.\n");
  barrier_init(&barrier, 2);

  dispatch_queue_t q =
      dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);

  // Case 1: Worker thread drops its block reference first (4 -> 2), then main
  // thread exits scope and drops the final reference (2 -> 1).
  {
    __block int x = 0;
    __block MyStruct s;

    // Hold as void* so the outer dispatch_async block does not retain it a
    // second time.
    void *ctx = _Block_copy((__bridge const void *)^{
      x = 42;
      (void)x;
      s.setGlobal();
    });

    dispatch_async(q, ^{
      ((__bridge void (^)(void))ctx)();
      _Block_release(ctx);
      barrier_wait(&barrier);
    });

    barrier_wait(&barrier);
  }

  // Case 2: Main thread writes to the __block variables after dispatch_async,
  // exits scope and drops its reference first (4 -> 2), and then the worker
  // thread drops the final reference (2 -> 1).
  barrier_init(&barrier, 2);
  barrier_init(&barrier2, 2);
  {
    __block int y = 0;
    __block MyStruct s2;

    // Note: we never execute this block - we just need to get the compiler to
    //       capture `y` and `s2`
    void *ctx2 = _Block_copy((__bridge const void *)^{
      (void)y;
      (void)s2;
    });

    dispatch_async(q, ^{
      barrier_wait(&barrier);
      _Block_release(ctx2);
      barrier_wait(&barrier2);
    });

    y = 1;
    s2.setGlobal();
  }
  barrier_wait(&barrier);
  barrier_wait(&barrier2);

  fprintf(stderr, "Done.\n");
  return 0;
}

// CHECK: Hello world.
// CHECK: Done.
