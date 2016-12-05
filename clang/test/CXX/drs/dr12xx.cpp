// RUN: %clang_cc1 -std=c++98 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++1z %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

// expected-no-diagnostics

namespace dr1213 { // dr1213: 4.0
#if __cplusplus >= 201103L
  using T = int[3];
  int &&r = T{}[1];

  using T = decltype((T{}));
  using U = decltype((T{}[2]));
  using U = int &&;
#endif
}

namespace dr1250 {  // dr1250: 3.9
struct Incomplete;

struct Base {
  virtual const Incomplete *meow() = 0;
};

struct Derived : Base {
  virtual Incomplete *meow();
};
} // dr1250
