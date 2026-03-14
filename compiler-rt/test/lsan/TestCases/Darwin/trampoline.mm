// Test that the memory region  that contains Objective-C block trampolines
// is scanned.
// FIXME: Find a way to reduce this without AppKit to remove Mac requirement.
// UNSUPPORTED: ios
// RUN: %clangxx_lsan %s  -o %t -framework Cocoa -fno-objc-arc
// RUN: %env_lsan_opts="report_objects=1" %run %t 2>&1  && echo "" | FileCheck %s

#import <Cocoa/Cocoa.h>

#include <sanitizer/lsan_interface.h>

int main() {
  NSView *view =
      [[[NSView alloc] initWithFrame:CGRectMake(0, 0, 20, 20)] autorelease];
  __lsan_do_leak_check();
  return 0;
}
// CHECK-NOT: LeakSanitizer: detected memory leaks
