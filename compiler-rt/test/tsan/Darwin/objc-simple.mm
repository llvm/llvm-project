// Test that a simple Obj-C program runs and exits without any warnings.

// RUN: %clang_tsan %s -o %t -framework Foundation
// RUN: %env_tsan_opts=ignore_interceptors_accesses=1 %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>

int main() {
  NSLog(@"Hello world");
}

// CHECK: Hello world
// CHECK-NOT: WARNING: ThreadSanitizer
