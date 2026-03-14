// RUN: %clang_cc1 -std=c++11 -triple=x86_64-linux-gnu -verify %s

struct S {
  static thread_local int a;
  static int b; // expected-note {{here}}
  thread_local int c; // expected-error {{'thread_local' is only allowed on variable declarations}}
  static thread_local int d; // expected-note {{here}}
};

thread_local int S::a;
thread_local int S::b; // expected-error {{thread-local declaration of 'b' follows non-thread-local declaration}}
thread_local int S::c; // expected-error {{non-static data member defined out-of-line}}
int S::d; // expected-error {{non-thread-local declaration of 'd' follows thread-local declaration}}

thread_local int x[3];
thread_local int y[3];
thread_local int z[3]; // expected-note {{previous}}

void f() {
  thread_local int x;
  static thread_local int y;
  extern thread_local int z; // expected-error {{redeclaration of 'z' with a different type}}
}
