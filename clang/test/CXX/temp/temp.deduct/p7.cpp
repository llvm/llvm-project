// RUN:  %clang_cc1 -verify %s

struct A {
  static constexpr bool x = true;
};

template<typename T, typename U>
void f(T, U) noexcept(T::x);

template<typename T, typename U>
void f(T, U*) noexcept(T::y); // expected-error {{no member named 'y' in 'A'}}

template<>
void f<A>(A, int*); // expected-note {{in instantiation of exception specification}}
