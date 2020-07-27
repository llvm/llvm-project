// RUN: %clang_cc1 -fsyntax-only -verify -Wunneeded-member-function -Wno-unused-template %s

namespace {
  class A {
    void g() {} // expected-warning {{member function 'g' is not needed and will not be emitted}}
    template <typename T> void gt(T) {}
    template <> void gt<int>(int) {} // expected-warning {{member function 'gt' is not needed and will not be emitted}}
    template <> void gt(float) {}    // expected-warning {{member function 'gt' is not needed and will not be emitted}}

    template <typename T>
    void foo() {
      g();
      gt(0);
      gt(0.0f);
      gt(0.0);
    }
  };
  template void A::gt(double); // no-warning
}
