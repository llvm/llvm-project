// RUN: not %clang -target x86_64-apple-darwin9 -fsyntax-only %s 2>&1 | FileCheck %s

// CHECK: error:
// CHECK-SAME: 'f0' is unavailable: introduced in macOS 11
// CHECK-NOT: unknown 

void f0(void) __attribute__((availability(macosx,strict,introduced=11)));

void client(void) {
f0(); }
