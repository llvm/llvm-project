// RUN: %clang_cc1 -fsyntax-only -verify %s
template<typename T>
void f(T);

template<typename T> // expected-note {{template parameter is declared here}}
struct A { };

struct X {
  template<> friend void f<int>(int); // expected-error{{in a friend}}
  template<> friend class A<int>; // expected-error{{cannot be a friend}}

  friend void f<float>(float); // okay
  friend class A<float>; // okay
};

struct PR41792 {
  // expected-error@+1{{cannot declare an explicit specialization in a friend}}
  template <> friend void f<>(int);

  // expected-error@+2{{template specialization declaration cannot be a friend}}
  // expected-error@+1{{missing template argument for template parameter}}
  template <> friend class A<>;
};
