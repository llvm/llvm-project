// RUN: %clang_tsan %s -o %t -framework Foundation
// RUN: %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>

long global;

void handler(void *arg) {
  fprintf(stderr, "global = %ld\n", global);

  dispatch_sync(dispatch_get_main_queue(), ^{
    CFRunLoopStop(CFRunLoopGetCurrent());
  });
}

int main(int argc, const char *argv[]) {
  dispatch_queue_t queue =
      dispatch_queue_create("my.queue", DISPATCH_QUEUE_CONCURRENT);

  dispatch_source_t source =
      dispatch_source_create(DISPATCH_SOURCE_TYPE_SIGNAL, SIGHUP, 0, queue);

  global = 42;

  dispatch_source_set_registration_handler_f(source, handler);

  dispatch_resume(source);

  CFRunLoopRun();

  return 0;
}

// CHECK: global = 42
// CHECK-NOT: WARNING: ThreadSanitizer
