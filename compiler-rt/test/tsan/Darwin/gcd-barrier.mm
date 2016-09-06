// RUN: %clang_tsan %s -o %t -framework Foundation
// RUN: %env_tsan_opts=ignore_interceptors_accesses=1 %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>

#import "../test.h"

long global;

int main() {
  fprintf(stderr, "Hello world.\n");
  print_address("addr=", 1, &global);
  barrier_init(&barrier, 2);

  dispatch_queue_t q = dispatch_queue_create("my.queue", DISPATCH_QUEUE_CONCURRENT);
  dispatch_queue_t bgq = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);

  dispatch_async(bgq, ^{
    dispatch_sync(q, ^{
      global = 42;
    });
    barrier_wait(&barrier);
  });

  dispatch_async(bgq, ^{
    barrier_wait(&barrier);
    dispatch_barrier_sync(q, ^{
      global = 43;
    });

    dispatch_async(bgq, ^{
      barrier_wait(&barrier);
      global = 44;
    });

    barrier_wait(&barrier);
    
    dispatch_sync(dispatch_get_main_queue(), ^{
      CFRunLoopStop(CFRunLoopGetCurrent());
    });
  });

  CFRunLoopRun();
  fprintf(stderr, "Done.\n");
}

// CHECK: Hello world.
// CHECK: Done.
// CHECK-NOT: WARNING: ThreadSanitizer
