// RUN: %clangxx_tsan %s -o %t -framework Foundation -fobjc-arc
// RUN: %run %t 2>&1 | FileCheck %s --implicit-check-not='ThreadSanitizer'

// @synchronized should not record a shadow read on the first byte of the
// object, otherwise it races with __tsan_external_write annotations that
// also access the same object.

#include "../test.h"

#import <Foundation/Foundation.h>
#import <sanitizer/tsan_interface.h>
#import <unistd.h>

__attribute__((noinline)) static void ExternalWrite(id obj, void *tag) {
  __tsan_external_write((__bridge void *)obj, __builtin_return_address(0), tag);
}

int main() {
  @autoreleasepool {
    barrier_init(&barrier, 2);
    void *tag = __tsan_external_register_tag("MyExternalObject");
    NSObject *obj = [NSObject new];
    __tsan_external_assign_tag((__bridge void *)obj, tag);
    dispatch_queue_t q =
        dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
    dispatch_apply(2, q, ^(size_t i) {
      if (i == 1)
        barrier_wait(&barrier);

      @synchronized(obj) {
        ExternalWrite(obj, tag);
      }

      if (i == 0)
        barrier_wait(&barrier);
    });
  }
  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK: DONE
