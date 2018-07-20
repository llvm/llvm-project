// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: cp %s %t
// RUN: not %clang_cc1 -fixit %t -x c -DFIXIT
// RUN: %clang_cc1 -fsyntax-only %t -x c -DFIXIT
// RUN: not %clang_cc1 -fsyntax-only -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s -strict-whitespace

void test1() {
  int a[] = {0,1,1,2,3};
  int []b = {0,1,4,9,16};
  // expected-error@-1{{brackets are not allowed here; to declare an array, place the brackets after the identifier}}
  // CHECK: {{^}}  int []b = {0,1,4,9,16};
  // CHECK: {{^}}      ~~ ^
  // CHECK: {{^}}         []
  // CHECK: fix-it:{{.*}}:{[[@LINE-5]]:7-[[@LINE-5]]:9}:""
  // CHECK: fix-it:{{.*}}:{[[@LINE-6]]:10-[[@LINE-6]]:10}:"[]"

  int c = a[0];
  int d = b[0]; // No undeclared identifier error here.

  int *e = a;
  int *f = b; // No undeclared identifier error here.
}

struct S {
  int [1][1]x;
  // expected-error@-1{{brackets are not allowed here; to declare an array, place the brackets after the identifier}}
  // CHECK: {{^}}  int [1][1]x;
  // CHECK: {{^}}      ~~~~~~ ^
  // CHECK: {{^}}             [1][1]
  // CHECK: fix-it:{{.*}}:{[[@LINE-5]]:7-[[@LINE-5]]:13}:""
  // CHECK: fix-it:{{.*}}:{[[@LINE-6]]:14-[[@LINE-6]]:14}:"[1][1]"
} s;

#ifndef FIXIT
void test2() {
  int [][][];
  // expected-error@-1{{expected identifier or '('}}
  // CHECK: {{^}}  int [][][];
  // CHECK: {{^}}      ^
  // CHECK-NOT: fix-it
  struct T {
    int [];
    // expected-error@-1{{expected member name or ';' after declaration specifiers}}
    // CHECK: {{^}}    int [];
    // CHECK: {{^}}    ~~~ ^
    // CHECK-NOT: fix-it
  };
}

void test3() {
  int [5] *;
  // expected-error@-1{{expected identifier or '('}}
  // CHECK: {{^}}  int [5] *;
  // CHECK: {{^}}           ^
  // CHECK-NOT: fix-it
  // expected-error@-5{{brackets are not allowed here; to declare an array, place the brackets after the identifier}}
  // CHECK: {{^}}  int [5] *;
  // CHECK: {{^}}      ~~~~ ^
  // CHECK: {{^}}          ()[5]
  // CHECK: fix-it:{{.*}}:{[[@LINE-9]]:7-[[@LINE-9]]:11}:""
  // CHECK: fix-it:{{.*}}:{[[@LINE-10]]:11-[[@LINE-10]]:11}:"("
  // CHECK: fix-it:{{.*}}:{[[@LINE-11]]:12-[[@LINE-11]]:12}:")[5]"

  int [5] * a;
  // expected-error@-1{{brackets are not allowed here; to declare an array, place the brackets after the identifier}}
  // CHECK: {{^}}  int [5] * a;
  // CHECK: {{^}}      ~~~~   ^
  // CHECK: {{^}}          (  )[5]
  // CHECK: fix-it:{{.*}}:{[[@LINE-5]]:7-[[@LINE-5]]:11}:""
  // CHECK: fix-it:{{.*}}:{[[@LINE-6]]:11-[[@LINE-6]]:11}:"("
  // CHECK: fix-it:{{.*}}:{[[@LINE-7]]:14-[[@LINE-7]]:14}:")[5]"

  int *b[5] = a;  // expected-error{{}} a should not be corrected to type b

  int (*c)[5] = a;  // a should be the same type as c
}
#endif

// CHECK: 8 errors generated.
