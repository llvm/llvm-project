// RUN: %clangxx_tsan %s -o %t -framework Foundation
// RUN: %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>

#import <atomic>
#import <cassert>
#import <cstdio>
#import <memory>

std::atomic<long> shared_call_counter(0);
std::atomic<long> weak_call_counter(0);
std::atomic<long> destructor_counter(0);
std::atomic<long> weak_destroyed_counter(0);

struct MyStruct {
  std::atomic<long> self_counter;
  MyStruct() : self_counter(0) { }
  virtual void shared_call() {
    std::atomic_fetch_add_explicit(&self_counter, 1, std::memory_order_relaxed);
    std::atomic_fetch_add_explicit(&shared_call_counter, 1, std::memory_order_relaxed);
  }
  virtual void weak_call() {
    std::atomic_fetch_add_explicit(&weak_call_counter, 1, std::memory_order_relaxed);
  }
  virtual ~MyStruct() {
    long n = self_counter;
    assert(n == 1000);
    std::atomic_fetch_add_explicit(&destructor_counter, 1, std::memory_order_relaxed);
  }
};

int main(int argc, const char *argv[]) {
  std::fprintf(stderr, "Hello world.\n");

  dispatch_queue_t q = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);

  dispatch_group_t g = dispatch_group_create();

  for (int i = 0; i < 1000; i++) {
    std::shared_ptr<MyStruct> shared(new MyStruct());
    std::weak_ptr<MyStruct> weak(shared);

    dispatch_group_async(g, q, ^{
      for (int j = 0; j < 1000; j++) {
        std::shared_ptr<MyStruct> shared_copy(shared);
        shared_copy->shared_call();
      }
    });
    dispatch_group_async(g, q, ^{
      for (int j = 0; j < 1000; j++) {
        std::shared_ptr<MyStruct> weak_copy = weak.lock();
        if (weak_copy) {
          weak_copy->weak_call();
        } else {
          std::atomic_fetch_add_explicit(&weak_destroyed_counter, 1, std::memory_order_relaxed);
          break;
        }
      }
    });
  }

  dispatch_group_wait(g, DISPATCH_TIME_FOREVER);

  std::fprintf(stderr, "shared_call_counter = %ld\n", shared_call_counter.load());
  std::fprintf(stderr, "weak_call_counter = %ld\n", weak_call_counter.load());
  std::fprintf(stderr, "destructor_counter = %ld\n", destructor_counter.load());
  std::fprintf(stderr, "weak_destroyed_counter = %ld\n", weak_destroyed_counter.load());

  std::fprintf(stderr, "Done.\n");
}

// CHECK: Hello world.
// CHECK: shared_call_counter = 1000000
// CHECK: destructor_counter = 1000
// CHECK: Done.
// CHECK-NOT: WARNING: ThreadSanitizer
