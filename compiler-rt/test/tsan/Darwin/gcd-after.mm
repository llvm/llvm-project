// RUN: %clang_tsan %s -o %t -framework Foundation
// RUN: %env_tsan_opts=ignore_interceptors_accesses=1 %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>

long my_global;
long my_global2;

void callback(void *context) {
  my_global2 = 42;

  dispatch_async(dispatch_get_main_queue(), ^{
    CFRunLoopStop(CFRunLoopGetMain());
  });
}

int main(int argc, const char *argv[]) {
  fprintf(stderr, "start\n");

  my_global = 10;
  dispatch_queue_t q = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
  dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(10 * NSEC_PER_MSEC)), q, ^{
    my_global = 42;

    dispatch_async(dispatch_get_main_queue(), ^{
      CFRunLoopStop(CFRunLoopGetMain());
    });
  });
  CFRunLoopRun();

  my_global2 = 10;
  dispatch_after_f(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(10 * NSEC_PER_MSEC)), q, NULL, &callback);
  CFRunLoopRun();

  fprintf(stderr, "done\n");
  return 0;
}

// CHECK: start
// CHECK: done
// CHECK-NOT: WARNING: ThreadSanitizer
