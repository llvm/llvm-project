// RUN: %clang_cc1 -emit-llvm-only -std=c++11 -g %s

namespace PR9414 {
  int f() {
    auto x = 0;
    return x;
  }
}

// Don't crash.
namespace PR13570 {
  template<typename T, typename U> struct P {};
  template<typename T> struct A {
    template<typename U> static P<T,U> isa(U);
    decltype(isa(int())) f() {}
  };
  template struct A<int>;
}
