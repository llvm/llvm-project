// Test that a simple Obj-C program runs and exits without any warnings.

// RUN: %clang_tsan %s -o %t -framework Foundation
// RUN: %run %t 2>&1

#import <Foundation/Foundation.h>

int main() {
  NSLog(@"Hello world");
}

// CHECK: Hello world
// CHECK-NOT: WARNING: ThreadSanitizer
