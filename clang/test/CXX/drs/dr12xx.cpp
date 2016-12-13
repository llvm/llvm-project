// RUN: %clang_cc1 -std=c++98 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++1z %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

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

namespace dr1295 {  // dr1295: 4.0
  struct X {
    unsigned bitfield : 4;
  };

  X x = {1};

  unsigned const &r1 = static_cast<X &&>(x).bitfield; // expected-error 0-1{{C++11}}
  unsigned const &r2 = static_cast<unsigned &&>(x.bitfield); // expected-error 0-1{{C++11}}

  template<unsigned &r> struct Y {};
  Y<x.bitfield> y;
#if __cplusplus <= 201402L
  // expected-error@-2 {{does not refer to any declaration}} expected-note@-3 {{here}}
#else
  // expected-error@-4 {{refers to subobject}}
#endif

#if __cplusplus >= 201103L
  const unsigned other = 0;
  using T = decltype(true ? other : x.bitfield);
  using T = unsigned;
#endif
}

