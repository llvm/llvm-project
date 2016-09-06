// RUN: %clang_tsan %s -o %t -framework Foundation
// RUN: %env_tsan_opts=ignore_interceptors_accesses=1 %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>

long global;

int main(int argc, const char *argv[]) {
  dispatch_queue_t queue =
      dispatch_queue_create("my.queue", DISPATCH_QUEUE_CONCURRENT);

  dispatch_source_t source =
      dispatch_source_create(DISPATCH_SOURCE_TYPE_SIGNAL, SIGHUP, 0, queue);

  global = 42;

  dispatch_source_set_registration_handler(source, ^{
    fprintf(stderr, "global = %ld\n", global);

    dispatch_sync(dispatch_get_main_queue(), ^{
      CFRunLoopStop(CFRunLoopGetCurrent());
    });
  });

  dispatch_resume(source);

  CFRunLoopRun();

  return 0;
}

// CHECK: global = 42
// CHECK-NOT: WARNING: ThreadSanitizer
