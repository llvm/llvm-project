// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace N0 {
  struct A {
    void f0() noexcept(x);
    void g0() noexcept(y); // expected-error {{use of undeclared identifier 'y'}}

    void f1() noexcept(A::x);
    void g1() noexcept(A::y); // expected-error {{no member named 'y' in 'N0::A'}}

    template<typename T>
    void f2() noexcept(x);
    template<typename T>
    void g2() noexcept(y); // expected-error {{use of undeclared identifier 'y'}}

    template<typename T>
    void f3() noexcept(A::x);
    template<typename T>
    void g3() noexcept(A::y); // expected-error {{no member named 'y' in 'N0::A'}}

    friend void f4() noexcept(x);
    friend void g4() noexcept(y); // expected-error {{use of undeclared identifier 'y'}}

    friend void f5() noexcept(A::x);
    friend void g5() noexcept(A::y); // expected-error {{no member named 'y' in 'N0::A'}}

    template<typename T>
    friend void f6() noexcept(x);
    template<typename T>
    friend void g6() noexcept(y); // expected-error {{use of undeclared identifier 'y'}}

    template<typename T>
    friend void f7() noexcept(A::x);
    template<typename T>
    friend void g7() noexcept(A::y); // expected-error {{no member named 'y' in 'N0::A'}}

    static constexpr bool x = true;
  };
} // namespace N0

namespace N1 {
  template<typename T>
  struct A {
    void f0() noexcept(x);
    void g0() noexcept(y); // expected-error {{use of undeclared identifier 'y'}}

    void f1() noexcept(A::x);
    void g1() noexcept(A::y); // expected-error {{no member named 'y' in 'A<T>'}}

    template<typename U>
    void f2() noexcept(x);
    template<typename U>
    void g2() noexcept(y); // expected-error {{use of undeclared identifier 'y'}}

    template<typename U>
    void f3() noexcept(A::x);
    template<typename U>
    void g3() noexcept(A::y); // expected-error {{no member named 'y' in 'A<T>'}}

    friend void f4() noexcept(x);
    friend void g4() noexcept(y); // expected-error {{use of undeclared identifier 'y'}}

    friend void f5() noexcept(A::x);
    friend void g5() noexcept(A::y); // expected-error {{no member named 'y' in 'A<T>'}}

    template<typename U>
    friend void f6() noexcept(x);
    template<typename U>
    friend void g6() noexcept(y); // expected-error {{use of undeclared identifier 'y'}}

    template<typename U>
    friend void f7() noexcept(A::x);
    template<typename U>
    friend void g7() noexcept(A::y); // expected-error {{no member named 'y' in 'A<T>'}}

    static constexpr bool x = true;
  };
} // namespace N1
