// RUN: c-index-test -test-load-source local %s 2>&1 | FileCheck %s

void test() {                              
  goto exit;
}

int foo;

#define NO 0

void f(int y) {
  if (y = NO);
}

int

// CHECK: cindex-on-invalid.m:4:8: error: use of undeclared label 'exit'
// CHECK: cindex-on-invalid.m:12:9:{12:7-12:13}
// CHECK: cindex-on-invalid.m:20:1: error: expected identifier or '('

