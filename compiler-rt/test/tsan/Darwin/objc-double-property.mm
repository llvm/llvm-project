// RUN: %clangxx_tsan -O0 %s -o %t -framework Foundation && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_tsan -O1 %s -o %t -framework Foundation && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_tsan -O2 %s -o %t -framework Foundation && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_tsan -O3 %s -o %t -framework Foundation && %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>

@interface MyClass : NSObject
@property float a;
@property double b;
@property long double c;
@end

@implementation MyClass
@end

int main() {
  NSLog(@"Hello world");
}

// CHECK: Hello world
