// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s

namespace GH49093 {
  class B {
  public:
    static int a() { return 0; }
    decltype(a< 0 >(0)) test; // expected-error {{'a' does not name a template but is followed by template arguments}}
  };

  struct C {
      static int a() { return 0; }
      decltype(a < 0 > (0)) test; // expected-error {{'a' does not name a template but is followed by template arguments}}
  };

  void test_is_bool(bool t) {}
  void test_is_bool(int t) {}

  int main() {
    B b;
    test_is_bool(b.test);

    C c;
    test_is_bool(c.test);
  }
}

namespace GH107047 {
  struct A {
    static constexpr auto test() { return 1; }
    static constexpr int s = test< 1 >(); // expected-error {{'test' does not name a template but is followed by template arguments}}
  };
}
