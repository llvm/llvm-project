// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

struct rdar9677163 {
  struct Y { ~Y(); };
  struct Z { ~Z(); };
  Y::~Y() { } // expected-error{{non-friend class member '~Y' cannot have a qualified name}}
  ~Z(); // expected-error{{expected the class name after '~' to name the enclosing class}}
};
