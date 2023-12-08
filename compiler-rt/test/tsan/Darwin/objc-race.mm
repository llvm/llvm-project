// RUN: %clang_tsan %s -o %t -framework Foundation
// RUN: %deflake %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>

#import "../test.h"

@interface MyClass : NSObject {
  long instance_variable;
}
- (void)method:(long)value;
@end

@implementation MyClass

- (void)method:(long)value {
  self->instance_variable = value;
}

@end

int main() {
  NSLog(@"Hello world.");
  barrier_init(&barrier, 2);
  
  MyClass *my_object = [MyClass new];
  [my_object method:42];
  
  dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
    [my_object method:43];
    barrier_wait(&barrier);
  });
  
  dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
    barrier_wait(&barrier);
    [my_object method:44];

    dispatch_sync(dispatch_get_main_queue(), ^{
      CFRunLoopStop(CFRunLoopGetCurrent());
    });
  });
  
  CFRunLoopRun();
  NSLog(@"Done.");
  return 0;
}

// CHECK: Hello world.
// CHECK: WARNING: ThreadSanitizer: data race
// CHECK:   Write of size 8
// CHECK:     #0 -[MyClass method:]
// CHECK:   Previous write of size 8
// CHECK:     #0 -[MyClass method:]
// CHECK:   Location is heap block
// CHECK: Done.
