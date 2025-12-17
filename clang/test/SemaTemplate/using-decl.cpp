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

namespace UsingUsingEnum {
  namespace foo {
    enum class EnumOne {};
  }
  using foo::EnumOne;

  template <class> void t() {
    using enum EnumOne;
  }
  template void t<void>();
} // namespace UsingUsingEnum
