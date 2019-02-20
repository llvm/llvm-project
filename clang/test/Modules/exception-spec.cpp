// RUN: %clang_cc1 -x c++ -std=c++17 -fmodules -fmodules-local-submodule-visibility -fmodules-cache-path=%t %s -verify

// expected-no-diagnostics

#pragma clang module build PR38627
module PR38627 {}
#pragma clang module contents
#pragma clang module begin PR38627
namespace PR38627 {
struct X {
  virtual ~X() {}
  struct C {
    friend X::~X();
  } c;
};
}
#pragma clang module end
#pragma clang module endbuild

#pragma clang module import PR38627

namespace PR38627 {
struct Y {
  virtual ~Y() {}
  struct C {
    friend Y::~Y();
  } c;
};
static_assert(noexcept(X().~X()));
static_assert(noexcept(Y().~Y()));

struct A { virtual ~A() = default; };
struct B : public A, public X {
  virtual ~B() override = default;
};
} // PR38627
