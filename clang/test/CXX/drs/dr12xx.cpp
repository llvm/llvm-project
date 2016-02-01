// RUN: %clang_cc1 -std=c++98 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++1z %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

// expected-no-diagnostics

namespace dr1250 {  // dr1250: 3.9
struct Incomplete;

struct Base {
  virtual const Incomplete *meow() = 0;
};

struct Derived : Base {
  virtual Incomplete *meow();
};
} // dr1250
