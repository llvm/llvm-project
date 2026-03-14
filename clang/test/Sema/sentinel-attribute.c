// RUN: %clang_cc1 -fsyntax-only -verify -Wno-strict-prototypes %s
// RUN: not %clang_cc1 -fsyntax-only -Wno-strict-prototypes %s -fdiagnostics-parseable-fixits 2>&1 | \
// RUN:         FileCheck %s --check-prefix=C
// RUN: not %clang_cc1 -fsyntax-only -Wno-strict-prototypes %s -fdiagnostics-parseable-fixits -x c++ -std=c++11 2>&1 | \
// RUN:         FileCheck %s --check-prefix=CXX11
int x __attribute__((sentinel)); //expected-warning{{'sentinel' attribute only applies to functions, methods and blocks}}

void f1(int a, ...) __attribute__ ((sentinel)); // expected-note {{function has been explicitly marked sentinel here}}
void f2(int a, ...) __attribute__ ((sentinel(1)));

void f3(int a, ...) __attribute__ ((sentinel("hello"))); //expected-error{{'sentinel' attribute requires parameter 1 to be an integer constant}}
void f4(int a, ...) __attribute__ ((sentinel(1, 2, 3))); //expected-error{{'sentinel' attribute takes no more than 2 arguments}}
void f4(int a, ...) __attribute__ ((sentinel(-1))); //expected-error{{parameter 1 less than zero}}
void f4(int a, ...) __attribute__ ((sentinel(0, 2))); // expected-error{{parameter 2 not 0 or 1}}

void f5(int a) __attribute__ ((sentinel)); //expected-warning{{'sentinel' attribute only supported for variadic functions}}


void f6() __attribute__((__sentinel__));  // expected-warning {{'sentinel' attribute requires named arguments}}

void g(void) {
  // The integer literal zero is not a sentinel.
  f1(1, 0); // expected-warning {{missing sentinel in function call}}
// C: fix-it:{{.*}}:{23:10-23:10}:", (void*) 0"
// CXX11: fix-it:{{.*}}:{23:10-23:10}:", nullptr"
}
