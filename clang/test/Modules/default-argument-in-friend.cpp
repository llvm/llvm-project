// RUN: %clang_cc1 -fmodules -std=c++20 -verify -x c++-module-map -fmodule-name=A %s

module A {
  module Declare {}
  module Friend {}
  module Redeclare {}
}

#pragma clang module contents

// First submodule: introduce a default argument.
#pragma clang module begin A.Declare
void f(int = 0);
#pragma clang module end

// Second submodule: extend redeclaration chain with an instantiated friend and
// then a non-friend. Both inherit the default argument.
#pragma clang module begin A.Friend
#pragma clang module import A.Declare
template<typename T> struct X {
  friend void f(int);
  using type = T;
};
using Y = X<int>::type;
void f(int);
#pragma clang module end

// Third submodule: redefine the default argument. This should be valid; the
// instantiated friend should not count as introducing a prior default argument.
#pragma clang module begin A.Redeclare
// expected-no-diagnostics
void f(int = 0);
#pragma clang module end
