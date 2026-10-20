// ObjC properties are atomic by default. For a struct that is too large to be
// accessed atomically, the synthesized accessors copy it under a lock that
// lives in the ObjC runtime. When not contended, the ObjC runtime can take
// that lock in a manner that is opaque to TSan. The unlock is tail-called, and
// thus appears to come from user code, and is not filtered by 
// ignore_noninstrumented_modules. As a result, TSan will 'see' an unlock of a
// never-locked lock. This test checks that this does not result in a false 
// positive report of "unlock of an unlocked mutex".

// RUN: %clangxx_tsan %s -o %t -framework Foundation -fobjc-arc
// RUN: %env_tsan_opts=detect_deadlocks=1:ignore_noninstrumented_modules=1 %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>

@interface Window : NSObject
@property CGRect frame;
@end

@implementation Window
@end

int main() {
  Window *window = [Window new];

  dispatch_apply(8, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                 ^(size_t worker) {
                   for (int i = 0; i < 100000; i++) {
                     CGRect frame = window.frame;
                     frame.origin.x += 1;
                     window.frame = frame;
                   }
                 });

  fprintf(stderr, "Done\n");
  return 0;
}

// CHECK-NOT: ThreadSanitizer
// CHECK: Done
