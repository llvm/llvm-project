// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace PR4441 {
  namespace A {
      struct B { };
      void operator+(B,B);
  }

  using A::operator+;
} // namespace PR4441

namespace qualified_name {
  namespace XXX {
    struct A {
      using type = int;
    };
  }

  namespace YYY {
    using XXX::A;
  }

  YYY::A::type x = nullptr;
  // expected-error@-1 {{variable of type 'YYY::A::type'}}
} // namespace qualifed_name
