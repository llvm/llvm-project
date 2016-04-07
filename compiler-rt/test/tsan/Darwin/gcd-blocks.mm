// RUN: %clang_tsan %s -o %t -framework Foundation
// RUN: %env_tsan_opts=ignore_interceptors_accesses=1 %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>

int main() {
  fprintf(stderr, "start\n");

  dispatch_queue_t background_q = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
  dispatch_queue_t main_q = dispatch_get_main_queue();

  dispatch_async(background_q, ^{
    __block long block_var = 0;

    dispatch_sync(main_q, ^{
      block_var = 42;
    });

    fprintf(stderr, "block_var = %ld\n", block_var);

    dispatch_sync(dispatch_get_main_queue(), ^{
      CFRunLoopStop(CFRunLoopGetCurrent());
    });
  });
  
  CFRunLoopRun();
  fprintf(stderr, "done\n");
}

// CHECK: start
// CHECK: block_var = 42
// CHECK: done
// CHECK-NOT: WARNING: ThreadSanitizer
// CHECK-NOT: CHECK failed
