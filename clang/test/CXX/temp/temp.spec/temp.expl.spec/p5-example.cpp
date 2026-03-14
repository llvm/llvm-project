// RUN: %clang_cc1 -fsyntax-only -verify %s
template<class T> struct A {
       struct B { };
       template<class U> struct C { };
};
     template<> struct A<int> {
       void f(int);
};
void h() {
  A<int> a;
  a.f(16);
}
// A<int>::f must be defined somewhere
// template<> not used for a member of an // explicitly specialized class template
void A<int>::f(int) { /* ... */ }
  template<> struct A<char>::B {
    void f();
};
// template<> also not used when defining a member of // an explicitly specialized member class
void A<char>::B::f() { /* ... */ }
  template<> template<class U> struct A<char>::C {
    void f();
};

template<>
template<class U> void A<char>::C<U>::f() { /* ... */ }
  template<> struct A<short>::B {
    void f();
};
template<> void A<short>::B::f() { /* ... */ } // expected-error{{no function template matches function template specialization 'f'}}
  template<> template<class U> struct A<short>::C {
    void f();
};
template<class U> void A<short>::C<U>::f() { /* ... */ } // expected-error{{template parameter list matching the non-templated nested type 'A<short>' should be empty ('template<>')}}
