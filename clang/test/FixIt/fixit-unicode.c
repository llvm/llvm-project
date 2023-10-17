// This file contains code and checks, that should work on any platform.
// There's a set of additional checks for systems with proper support of UTF-8
// on the standard output in fixit-unicode-with-utf8-output.c.

// RUN: not %clang_cc1 -fsyntax-only -fno-diagnostics-show-line-numbers %s 2>&1 | FileCheck -strict-whitespace %s
// RUN: not %clang_cc1 -fsyntax-only -fno-diagnostics-show-line-numbers -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck -check-prefix=CHECK-MACHINE %s

struct Foo {
  int bar;
};

// PR13312
void test1() {
  struct Foo foo;
  foo.bar = 42☃
  // CHECK: error: character <U+2603> not allowed in an identifier
  // CHECK: {{^              \^}}
  // Make sure we emit the fixit right in front of the snowman.

  // CHECK-MACHINE: fix-it:"{{.*}}":{[[@LINE-5]]:15-[[@LINE-5]]:18}:""
}


int printf(const char *, ...);
void test2() {
  printf("∆: %d", 1L);
// CHECK: warning: format specifies type 'int' but the argument has type 'long'
// Don't crash emitting a fixit after the delta.
// CHECK:  printf("
// CHECK: : %d", 1L);
// Unfortunately, we can't actually check the location of the printed fixit,
// because different systems will render the delta differently (either as a
// character, or as <U+2206>.) The fixit should line up with the %d regardless.

// CHECK-MACHINE: fix-it:"{{.*}}":{[[@LINE-9]]:16-[[@LINE-9]]:18}:"%ld"
}

void test3() {
  int กssss = 42;
  int a = กsss; // expected-error{{use of undeclared identifier 'กsss'; did you mean 'กssss'?}}
// CHECK: {{^          \^}}
// CHECK: {{^          [^ ]+ssss}}
// CHECK-MACHINE: fix-it:"{{.*}}":{[[@LINE-3]]:11-[[@LINE-3]]:17}:"\340\270\201ssss"

  int ssกss = 42;
  int b = ssกs; // expected-error{{use of undeclared identifier 'ssกs'; did you mean 'ssกss'?}}
// CHECK: {{^          \^}}
// CHECK: {{^          ss.+ss}}
// CHECK-MACHINE: fix-it:"{{.*}}":{[[@LINE-3]]:11-[[@LINE-3]]:17}:"ss\340\270\201ss"

  int s一二三 = 42;
  int b一二三四五六七 = ss一二三; // expected-error{{use of undeclared identifier 'ss一二三'; did you mean 's一二三'?}}
// CHECK-MACHINE: fix-it:"{{.*}}":{[[@LINE-1]]:32-[[@LINE-1]]:43}:"s\344\270\200\344\272\214\344\270\211"


  int sssssssssก = 42;
  int c = sssssssss; // expected-error{{use of undeclared identifier 'sssssssss'; did you mean 'sssssssssก'?}}
// CHECK: {{^          \^}}
// CHECK: {{^          sssssssss.+}}
// CHECK-MACHINE: fix-it:"{{.*}}":{[[@LINE-3]]:11-[[@LINE-3]]:20}:"sssssssss\340\270\201"
}
