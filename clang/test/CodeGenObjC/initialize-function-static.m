// RUN: %clang_cc1 -triple x86_64-apple-macos10.15 -emit-llvm -fobjc-arc -o - %s | FileCheck %s

@interface I
@end

I *i(void) {
  static I *i = ((void *)0);
  return i;
}

// CHECK-NOT: __cxa_guard_acquire
// CHECK-NOT: __cxa_guard_release
