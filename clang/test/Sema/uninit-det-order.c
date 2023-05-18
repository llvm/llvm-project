// RUN: %clang_cc1 -Wuninitialized -fsyntax-only %s 2>&1 | FileCheck %s

void pr14901(int a) {
   int b, c;
   a = b;
   a = c;
}

// CHECK: 5:8: warning: variable 'b' is uninitialized when used here
// CHECK: 4:9: note: initialize the variable 'b' to silence this warning
// CHECK: 6:8: warning: variable 'c' is uninitialized when used here
// CHECK: 4:12: note: initialize the variable 'c' to silence this warning

