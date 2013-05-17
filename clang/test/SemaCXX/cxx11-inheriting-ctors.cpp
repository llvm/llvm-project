// RUN: %clang_cc1 -std=c++11 %s -verify

// expected-no-diagnostics

namespace PR15757 {
  struct S {
  };

  template<typename X, typename Y> struct T {
    template<typename A> T(X x, A &&a) {}

    template<typename A> explicit T(A &&a)
        noexcept(noexcept(T(X(), static_cast<A &&>(a))))
      : T(X(), static_cast<A &&>(a)) {}
  };

  template<typename X, typename Y> struct U : T<X, Y> {
    using T<X, Y>::T;
  };

  U<S, char> foo(char ch) { return U<S, char>(ch); }

  int main() {
    U<S, int> a(42);
    U<S, char> b('4');
    return 0;
  }
}
