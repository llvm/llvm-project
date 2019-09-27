// RUN: %clang_cc1 -std=c++2a -verify %s
// expected-no-diagnostics

namespace UsingInGenericLambda {
  namespace a {
    enum { b };
  }
  template<typename> void c() {
    auto d = [](auto) {
      using a::b;
      (void)b;
    };
    d(0);
  }
  void e() { c<int>(); }
}
