// RUN: %clang_cc1 -triple x86_64-apple-macosx10.9.0 -std=c++11 -fsyntax-only -verify %s

__attribute__((availability(macos, introduced = 10.0))) int init10();
__attribute__((availability(macos, introduced = 11.0))) int init11(); // expected-note 2 {{'init11' has been marked as being introduced in macOS 11.0}}

struct B0 {
  B0(int);
};

struct B1 {
  B1(int);
};

struct S : B0, B1 {
  S() : B0(init10()),
        B1(init11()), // expected-warning {{'init11' is only available on macOS 11.0}} expected-note {{enclose 'init11'}}
        i0(init10()),
        i1(init11())  // expected-warning {{'init11' is only available on macOS 11.0}} expected-note {{enclose 'init11'}}
  {}
  int i0, i1;
};
