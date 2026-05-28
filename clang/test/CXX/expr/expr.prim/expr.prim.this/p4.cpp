// RUN: %clang_cc1 -std=c++23 -fsyntax-only -Wno-unused-value -verify %s

namespace N0 {
  struct A {
    static void f() {
      []() -> decltype(this) { }; // expected-error{{invalid use of 'this' outside of a non-static member function}}
    }
  };
} // namespace N0

namespace N1 {
  struct A {
    static void f() {
      []() noexcept(decltype(this)()) { }; // expected-error{{invalid use of 'this' outside of a non-static member function}}
    }
  };
} // namespace N1

namespace N2 {
  struct A {
    static void f() {
      [](this auto&&) -> decltype(this) { }; // expected-error{{invalid use of 'this' outside of a non-static member function}}
    }
  };
} // namespace N2

namespace N3 {
  struct A {
    static void f() {
      [](this auto&&) noexcept(decltype(this)()) { }; // expected-error{{invalid use of 'this' outside of a non-static member function}}
    }
  };
} // namespace N3

namespace N4 {
  struct A {
    void f() {
      []() -> decltype(this) { };
    }
  };
} // namespace N4

namespace N5 {
  struct A {
    void f() {
      []() noexcept(decltype(this)()) { }; // expected-error{{conversion from 'decltype(this)' (aka 'N5::A *') to 'bool' is not allowed in a converted constant expression}}
    }
  };
} // namespace N5

namespace N6 {
  struct A {
    void f() {
      [](this auto&&) -> decltype(this) { };
    }
  };
} // namespace N6

namespace N7 {
  struct A {
    void f() {
      [](this auto&&) noexcept(decltype(this)()) { }; // expected-error{{conversion from 'decltype(this)' (aka 'N7::A *') to 'bool' is not allowed in a converted constant expression}}
    }
  };
} // namespace N7
