// RUN: %clang_cc1 -triple x86_64-apple-darwin10  -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

@interface I
@end

void foo(void *p) {
  I *i = (__bridge_transfer I*)p;
  I *i2 = (__bridge_transfer/*cake*/I*)p;
}

// CHECK: {7:11-7:29}:""
// CHECK: {8:12-8:29}:""
