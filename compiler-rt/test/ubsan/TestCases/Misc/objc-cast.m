// REQUIRES: darwin
//
// RUN: %clang -framework Foundation -fsanitize=objc-cast %s -O1 -o %t
// RUN: %run %t 2>&1 | FileCheck %s
//
// RUN: %clang -framework Foundation -fsanitize=objc-cast -fno-sanitize-recover=objc-cast %s -O1 -o %t.trap
// RUN: not %run %t.trap 2>&1 | FileCheck %s

#include <Foundation/Foundation.h>

int main() {
  NSArray *arrayOfInt = [NSArray arrayWithObjects:@1, @2, @3, (void *)0];
  // CHECK: objc-cast.m:[[@LINE+1]]:{{.*}}: runtime error: invalid ObjC cast, object is a '__NSCFNumber', but expected a 'NSString'
  for (NSString *str in arrayOfInt) {
    NSLog(@"%@", str);
  }

  NSArray *arrayOfStr = [NSArray arrayWithObjects:@"a", @"b", @"c", (void *)0];
  for (NSString *str in arrayOfStr) {
    NSLog(@"%@", str);
  }

  // The diagnostic should only be printed once.
  // CHECK-NOT: runtime error

  return 0;
}
