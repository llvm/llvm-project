// RUN: %clang_cc1 -verify %s -std=c++11

template<typename T> struct A {
  template<typename U> struct B;
  template<typename U> using C = U; // expected-note {{here}}
};

struct X {
  template<typename T> X(T);
  struct Y {
    template<typename T> Y(T);
  };
};

template<typename T> A // expected-warning {{missing 'typename'}}
                      <T>::B<T> f1();
template<typename T> A<T>::C<T> f2(); // expected-warning {{missing 'typename'}}

// FIXME: Should these cases really be valid? There doesn't appear to be a rule prohibiting them...
template<typename T> A<T>::C<X>::X(T) {}
template<typename T> A<T>::C<X>::X::Y::Y(T) {}

// FIXME: This is ill-formed
template<typename T> int A<T>::B<T>::*f3() {}
template<typename T> int A<T>::C<X>::*f4() {}

// FIXME: This is valid
template<typename T> int A<T>::template C<int>::*f5() {} // expected-error {{has no members}}

template<typename T> template<typename U> struct A<T>::B {
  friend A<T>::C<T> f6(); // ok, same as 'friend T f6();'

  friend A<U>::C<T> f7(); // expected-error {{use 'template' keyword to treat 'C' as a dependent template name}} expected-warning {{missing 'typename'}}
  friend A<U>::template C<T> f8(); // expected-warning {{missing 'typename'}}
};
