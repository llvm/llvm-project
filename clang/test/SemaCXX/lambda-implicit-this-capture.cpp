// RUN: %clang_cc1 -std=c++11 -verify=cxx11 %s
// RUN: %clang_cc1 -std=c++2a -verify=cxx2a %s
// RUN: %clang_cc1 -std=c++2a -verify=cxx2a-no-deprecated %s -Wno-deprecated
// cxx11-no-diagnostics
// cxx2a-no-deprecated-no-diagnostics

struct A {
  int i;
  void f() {
    (void) [=] { // cxx2a-note {{add an explicit capture of 'this'}}
      return i; // cxx2a-warning {{implicit capture of 'this' with a capture default of '=' is deprecated}}
    };
  }
};
