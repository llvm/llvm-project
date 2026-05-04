// RUN: %clang_cc1 -verify -std=c++2a -pedantic-errors %s
// RUN: cp %s %t
// RUN: not %clang_cc1 -x c++ -std=c++2a -fixit %t
// RUN: %clang_cc1 -Wall -pedantic-errors -x c++ -std=c++2a %t
// RUN: cat %t | FileCheck %s

/* This is a test of the various code modification hints that only
   apply in C++2a. */
template<typename ...T> void init_capture_pack(T ...a) {
  [x... = a]{}; // expected-error {{must appear before the name}}
  [x = a...]{}; // expected-error {{must appear before the name}}
  [...&x = a]{}; // expected-error {{must appear before the name}}
  [...a]{}; // expected-error {{must appear after the name}}
  [&...a]{}; // expected-error {{must appear after the name}}
  [...&a]{}; // expected-error {{must appear after the name}}
}

namespace constinit_mismatch {
  int b = 123; // expected-note {{add the 'constinit' specifier}}
  extern constinit int b; // expected-error {{'constinit' specifier added after initialization of variable}}
  // CHECK: {{^}}  extern int b;

  template<typename> struct X {
    template<int> static constinit int n; // expected-note {{constinit}}
  };
  template<typename T> template<int N>
  int X<T>::n = 123; // expected-error {{missing}}
  // CHECK: {{^}}  constinit int X<T>::n = 123;

#define ABSL_CONST_INIT [[clang::require_constant_initialization]]
  extern constinit int c; // expected-note {{constinit}}
  int c; // expected-error {{missing}}
  // CHECK: {{^}}  ABSL_CONST_INIT int c;

#define MY_CONST_INIT constinit
  extern constinit int d; // expected-note {{constinit}}
  int d; // expected-error {{missing}}
  // CHECK: {{^}}  MY_CONST_INIT int d;
#undef MY_CONST_INIT

  extern constinit int e; // expected-note {{constinit}}
  int e; // expected-error {{missing}}
  // CHECK: {{^}}  ABSL_CONST_INIT int e;
#undef ABSL_CONST_INIT
}
