// Regression test for
// https://code.google.com/p/address-sanitizer/issues/detail?id=360.

// RUN: %clang_asan %s -o %t -framework Foundation
// RUN: %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>

void f() {
    int y = 7;
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_BACKGROUND, 0), ^{
        dispatch_sync(dispatch_get_main_queue(), ^{
            printf("num = %d\n", y);
        });
    });
}

int main() {
  NSLog(@"Hello world");
}

// CHECK-NOT: AddressSanitizer: odr-violation
// CHECK: Hello world
