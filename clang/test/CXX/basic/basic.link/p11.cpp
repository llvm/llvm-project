// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s

namespace MemberSpecialization {
  template<typename T>
  struct A {
    template<bool B>
    void f() noexcept(B);

    template<bool B>
    void g() noexcept(B); // expected-note {{previous declaration is here}}
  };

  template<>
  template<bool B>
  void A<int>::f() noexcept(B);

  template<>
  template<bool B>
  void A<int>::g() noexcept(!B); // expected-error {{exception specification in declaration does not match previous declaration}}
}

namespace Friend {
  template<bool B>
  void f() noexcept(B);

  template<bool B>
  void g() noexcept(B); // expected-note {{previous declaration is here}}

  template<typename T>
  struct A {
    template<bool B>
    friend void f() noexcept(B);

    template<bool B>
    friend void g() noexcept(!B); // expected-error {{exception specification in declaration does not match previous declaration}}
  };
}
