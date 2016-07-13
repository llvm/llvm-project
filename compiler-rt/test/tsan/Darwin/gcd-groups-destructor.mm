// RUN: %clangxx_tsan %s -o %t -framework Foundation
// RUN: %env_tsan_opts=ignore_interceptors_accesses=1 %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>

#import <memory>
#import <stdatomic.h>

_Atomic(long) destructor_counter = 0;

struct MyStruct {
  virtual ~MyStruct() {
    usleep(10000);
    atomic_fetch_add_explicit(&destructor_counter, 1, memory_order_relaxed);
  }
};

int main(int argc, const char *argv[]) {
  fprintf(stderr, "Hello world.\n");

  dispatch_queue_t q = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
  dispatch_group_t g = dispatch_group_create();

  for (int i = 0; i < 100; i++) {
    std::shared_ptr<MyStruct> shared(new MyStruct());

    dispatch_group_async(g, q, ^{
      shared.get(); // just to make sure the object is captured by the block
    });
  }

  dispatch_group_wait(g, DISPATCH_TIME_FOREVER);

  if (destructor_counter != 100) {
    abort();
  }

  fprintf(stderr, "Done.\n");
}

// CHECK: Hello world.
// CHECK-NOT: WARNING: ThreadSanitizer
// CHECK: Done.
