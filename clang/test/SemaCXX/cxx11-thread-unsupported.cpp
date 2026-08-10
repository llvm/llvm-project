// RUN: %clang_cc1 -std=c++11 -triple=x86_64-apple-macosx10.6 -verify %s

void f() {
  thread_local int x; // expected-error {{thread-local storage is not supported for the current target}}
}
