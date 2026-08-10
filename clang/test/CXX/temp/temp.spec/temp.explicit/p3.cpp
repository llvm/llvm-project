// RUN: %clang_cc1 -fsyntax-only -verify -Wc++11-compat %s

// A declaration of a function template shall be in scope at the point of the 
// explicit instantiation of the function template.
template<typename T> void f0(T);
template void f0(int); // okay
template<typename T> void f0(T) { }

// A definition of the class or class template containing a member function 
// template shall be in scope at the point of the explicit instantiation of 
// the member function template.
struct X0; // expected-note {{forward declaration}}
template<typename> struct X1; // expected-note 5{{declared here}}

template void X0::f0<int>(int); // expected-error {{incomplete type}}
template void X1<int>::f0<int>(int); // expected-error {{implicit instantiation of undefined template}}

// A definition of a class template or class member template shall be in scope 
// at the point of the explicit instantiation of the class template or class 
// member template.
template struct X1<float>; // expected-error{{explicit instantiation of undefined template}}

template<typename T>
struct X2 { // expected-note 4{{refers here}}
  template<typename U>
  struct Inner; // expected-note{{declared here}}
  
  struct InnerClass; // expected-note{{forward declaration}}
};

template struct X2<int>::Inner<float>; // expected-error{{explicit instantiation of undefined template}}

// A definition of a class template shall be in scope at the point of an 
// explicit instantiation of a member function or a static data member of the
// class template.
template void X1<int>::f1(int); // expected-error {{undefined template}}
template void X1<int>::f1<int>(int); // expected-error {{undefined template}}

template int X1<int>::member; // expected-error {{undefined template}}

// A definition of a member class of a class template shall be in scope at the 
// point of an explicit instantiation of the member class.
template struct X2<float>::InnerClass; // expected-error{{undefined member}}

// If the declaration of the explicit instantiation names an implicitly-declared 
// special member function (Clause 12), the program is ill-formed.
template X2<int>::X2(); // expected-error{{not an instantiation}}
template X2<int>::X2(const X2&); // expected-error{{not an instantiation}}
template X2<int>::~X2(); // expected-error{{not an instantiation}}
template X2<int> &X2<int>::operator=(const X2<int>&); // expected-error{{not an instantiation}}


// A definition of a class template is sufficient to explicitly
// instantiate a member of the class template which itself is not yet defined.
namespace PR7979 {
  template <typename T> struct S {
    void f();
    static void g();
    static int i;
    struct S2 {
      void h();
    };
  };

  template void S<int>::f();
  template void S<int>::g();
  template int S<int>::i;
  template void S<int>::S2::h();

  template <typename T> void S<T>::f() {}
  template <typename T> void S<T>::g() {}
  template <typename T> int S<T>::i;
  template <typename T> void S<T>::S2::h() {}
}

namespace PR11599 {
  template <typename STRING_TYPE> class BasicStringPiece;  // expected-note {{template is declared here}}

  extern template class BasicStringPiece<int>;  // expected-error{{explicit instantiation of undefined template 'PR11599::BasicStringPiece<int>}}
  template class BasicStringPiece<int>;
}
