// RUN: %clangxx_tsan %s -o %t -framework Foundation
// RUN: %env_tsan_opts=ignore_interceptors_accesses=1 %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>

#import <memory>

#import "../test.h"

long my_global;

struct MyStruct {
  void setGlobal() {
    my_global = 42;
  }
  ~MyStruct() {
    my_global = 43;
  }
};

int main(int argc, const char *argv[]) {
  fprintf(stderr, "Hello world.\n");
  print_address("addr=", 1, &my_global);
  barrier_init(&barrier, 2);

  std::shared_ptr<MyStruct> shared(new MyStruct());

  dispatch_queue_t q = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);

  std::weak_ptr<MyStruct> weak(shared);
  
  dispatch_async(q, ^{
    {
      std::shared_ptr<MyStruct> strong = weak.lock();
      if (!strong) exit(1);

      strong->setGlobal();
    }
    barrier_wait(&barrier);
  });

  barrier_wait(&barrier);
  shared.reset();

  fprintf(stderr, "Done.\n");
}

// CHECK: Hello world.
// CHECK: Done.
// CHECK-NOT: WARNING: ThreadSanitizer
