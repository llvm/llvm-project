// RUN: %clang_cc1 %s 2>&1 | FileCheck %s

void f(void) {
  // 600 sizeof's
  int a =
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof sizeof
  0;
  (void)a;

  // 600 of sizeof and __alignof
  int b =
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof sizeof __alignof
  0;
  (void)b;
}

// CHECK: warning: stack nearly exhausted; compilation time may suffer, and crashes due to stack overflow are likely
