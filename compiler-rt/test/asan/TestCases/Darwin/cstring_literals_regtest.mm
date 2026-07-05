// Regression test for
// https://code.google.com/p/address-sanitizer/issues/detail?id=274.

// RUN: %clang_asan %s -framework Foundation -o %t
// RUN: %run %t 2>&1 | FileCheck %s
#import <Foundation/Foundation.h>

#include <stdio.h>

int main() {
  NSString* version_file = @"MAJOR=35\n";
  int major = 0, minor = 0, build = 0, patch = 0;
  NSScanner* scanner = [NSScanner scannerWithString:version_file];
  NSString *res = nil;
  if ([scanner scanString:@"MAJOR=" intoString:nil] &&
      [scanner scanInt:&major]) {
    res = [NSString stringWithFormat:@"%d.%d.%d.%d",
           major, minor, build, patch];
  }
  printf("%s\n", [res UTF8String]);
  // CHECK: 35.0.0.0
  return 0;
}
