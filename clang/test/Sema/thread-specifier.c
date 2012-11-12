// RUN: %clang_cc1 -triple i686-pc-linux-gnu -fsyntax-only -Wno-private-extern -verify -pedantic %s

__thread int t1;
__thread extern int t2; // expected-warning {{'__thread' before 'extern'}}
__thread static int t3; // expected-warning {{'__thread' before 'static'}}
__thread __private_extern__ int t4;
struct t5 { __thread int x; }; // expected-error {{type name does not allow storage class to be specified}}
__thread int t6(); // expected-error {{'__thread' is only allowed on variable declarations}}

int f(__thread int t7) { // expected-error {{'__thread' is only allowed on variable declarations}}
  __thread int t8; // expected-error {{'__thread' variables must have global storage}}
  extern __thread int t9;
  static __thread int t10;
  __thread __private_extern__ int t11;
  __thread auto int t12; // expected-error {{'__thread' variables must have global storage}}
  __thread register int t13; // expected-error {{'__thread' variables must have global storage}}
}

__thread typedef int t14; // expected-error {{'__thread' is only allowed on variable declarations}}
__thread int t15; // expected-note {{previous definition is here}}
int t15; // expected-error {{non-thread-local declaration of 't15' follows thread-local declaration}}
int t16; // expected-note {{previous definition is here}}
__thread int t16; // expected-error {{thread-local declaration of 't16' follows non-thread-local declaration}}

// PR13720
__thread int thread_int;
int *thread_int_ptr = &thread_int; // expected-error{{initializer element is not a compile-time constant}}
void g() {
  int *p = &thread_int; // This is perfectly fine, though.
}
