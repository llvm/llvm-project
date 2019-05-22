// RUN: %clang_cc1 -verify -std=c++2a %s
// RUN: cp %s %t
// RUN: not %clang_cc1 -x c++ -std=c++2a -fixit %t
// RUN: %clang_cc1 -Wall -pedantic -x c++ -std=c++2a %t

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
