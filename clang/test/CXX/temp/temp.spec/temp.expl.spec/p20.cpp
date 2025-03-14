// RUN: %clang_cc1 -fsyntax-only -verify %s
template<typename T>
void f(T);

template<typename T>
struct A { }; // expected-note{{template is declared here}}

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
  // expected-error@+1{{too few template arguments for class template 'A'}}
  template <> friend class A<>;
};
