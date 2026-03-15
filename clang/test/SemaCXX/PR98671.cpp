// RUN: %clang_cc1 -std=c++20 -fsyntax-only %s -verify

struct S1 {
  operator int();

  template <typename T>
  operator T();
};


// Ensure that no assertion is raised when overload resolution fails while
// choosing between an operator function template and an operator function.
constexpr auto r = &S1::operator int;
// expected-error@-1 {{initializer of type '<overloaded function type>'}}


template <typename T>
struct S2 {
  template <typename U=T>
    S2(U={}) requires (sizeof(T) > 0) {}
    // expected-note@-1 {{candidate constructor}}

  template <typename U=T>
    S2(U={}) requires (true) {}
    // expected-note@-1 {{candidate constructor}}
};

S2<int> s;  // expected-error {{call to constructor of 'S2<int>' is ambiguous}}
