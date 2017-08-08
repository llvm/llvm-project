// RUN: %clang_cc1 -fsyntax-only -verify -std=c++1z %s

// C++1z [temp.local]p1:
//   Like normal (non-template) classes, class templates have an
//   injected-class-name (Clause 9). The injected-class-name can
//   be used as a template-name or a type-name.

template<typename> char id;

template<typename> struct TempType {};
template<template<typename> class> struct TempTemp {};

template<typename> void use(int&); // expected-note {{invalid explicitly-specified argument}} expected-note {{no known conversion}}
template<template<typename> class> void use(float&); // expected-note 2{{no known conversion}}
template<int> void use(char&); // expected-note 2{{invalid explicitly-specified argument}}

template<typename T> struct A {
  template<typename> struct C {};
  struct B : C<T> {
    //   When it is used with a template-argument-list,
    A<int> *aint;
    typename B::template C<int> *cint;

    //   as a template-argument for a template template-parameter,
    TempTemp<A> a_as_temp;
    TempTemp<B::template C> c_as_temp;

    //   or as the final identifier in the elaborated-type-specifier of a friend
    //   class template declaration,
    template<typename U> friend struct A;
    // it refers to the class template itself.

    // Otherwise, it is equivalent to the template-name followed by the
    // template-parameters of the class template enclosed in <>.
    A *aT;
    typename B::C *cT;
    TempType<A> a_as_type;
    TempType<typename B::C> c_as_type;
    friend struct A;
    friend struct B::C;

    void f(T &t) {
      use<A>(t); // expected-error {{no matching function}}
      if constexpr (&id<T> != &id<int>)
        use<B::template C>(t); // expected-error {{no matching function}}
    }
  };
};

template struct A<int>;
template struct A<float>;
template struct A<char>; // expected-note {{instantiation of}}

template <typename T> struct X0 {
  X0();
  ~X0();
  X0 f(const X0&);
};

// Test non-type template parameters.
template <int N1, const int& N2, const int* N3> struct X1 {
  X1();
  ~X1();
  X1 f(const X1& x1a) { X1 x1b(x1a); return x1b; }
};

//   When it is used with a template-argument-list, it refers to the specified
//   class template specialization, which could be the current specialization
//   or another specialization.
// FIXME: Test this clause.

int i = 42;
void test() {
  X0<int> x0; (void)x0;
  X1<42, i, &i> x1; (void)x1;
}
