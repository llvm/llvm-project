// RUN: %clang_cc1 -std=c++20 -verify %s

// Templates and partial and explicit specializations can't have C linkage.
namespace extern_c_templates {

template<typename T> struct A {
  static int a;
  struct b;
  void c();
  enum class d;

  template<typename U> static int e;
  template<typename U> struct f;
  template<typename U> void g();
};

template<typename T> int B;
template<typename T> void C();

extern "C" { // expected-note 1+{{begins here}}
  // templates
  template<typename T> struct A; // expected-error {{templates must have C++ linkage}}
  template<typename T> int B; // expected-error {{templates must have C++ linkage}}
  template<typename T> void C(); // expected-error {{templates must have C++ linkage}}

  // non-template members of a template
  // FIXME: Should these really be valid?
  template<typename T> int A<T>::a;
  template<typename T> struct A<T>::b {};
  template<typename T> void A<T>::c() {}
  template<typename T> enum class A<T>::d {};

  // templates
  template<typename T> template<typename U> int A<T>::e; // expected-error {{templates must have C++ linkage}}
  template<typename T> template<typename U> struct A<T>::f {}; // expected-error {{templates must have C++ linkage}}
  template<typename T> template<typename U> void A<T>::g() {} // expected-error {{templates must have C++ linkage}}

  // partial specializations
  template<typename T> struct A<int*>; // expected-error {{templates must have C++ linkage}}
  template<typename T> int B<int*>; // expected-error {{templates must have C++ linkage}}
  template<typename T> template<typename U> int A<T>::e<U*>; // expected-error {{templates must have C++ linkage}}
  template<typename T> template<typename U> struct A<T>::f<U*> {}; // expected-error {{templates must have C++ linkage}}

  // explicit specializations of templates
  template<> struct A<char> {}; // expected-error {{templates must have C++ linkage}}
  template<> int B<char>; // expected-error {{templates must have C++ linkage}}
  template<> void C<char>() {} // expected-error {{templates must have C++ linkage}}

  // explicit specializations of members of a template
  template<> int A<int>::a; // expected-error {{templates must have C++ linkage}}
  template<> struct A<int>::b {}; // expected-error {{templates must have C++ linkage}}
  template<> void A<int>::c() {} // expected-error {{templates must have C++ linkage}}
  template<> enum class A<int>::d {}; // expected-error {{templates must have C++ linkage}}

  // explicit specializations of member templates
  template<> template<typename U> int A<int>::e; // expected-error {{templates must have C++ linkage}}
  template<> template<typename U> struct A<int>::f {}; // expected-error {{templates must have C++ linkage}}
  template<> template<typename U> void A<int>::g() {} // expected-error {{templates must have C++ linkage}}
}

// Provide valid definitions for the explicit instantiations below.
// FIXME: Our recovery from the invalid definitions above isn't very good.
template<typename T> template<typename U> int A<T>::e;
template<typename T> template<typename U> struct A<T>::f {};
template<typename T> template<typename U> void A<T>::g() {}

extern "C" {
  // explicit instantiations
  // FIXME: Should these really be valid?
  template struct A<double>;
  template int A<float>::a;
  template struct A<float>::b;
  template void A<float>::c();
  template int A<float>::e<float>;
  template struct A<float>::f<float>;
  template void A<float>::g<float>();
}

}
