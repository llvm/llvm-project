// RUN: c-index-test -test-load-source local %s 2>&1 | FileCheck %s

// <rdar://problem/9123493>
void test() {                              
  goto exit;
}

int foo;

#define NO 0

void f(int y) {
  if (y = NO);
}

int

// CHECK: cindex-on-invalid.m:5:8: error: use of undeclared label 'exit'
// CHECK: cindex-on-invalid.m:13:9:{13:7-13:13}
// CHECK: cindex-on-invalid.m:21:1: error: expected identifier or '('

