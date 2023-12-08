// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

template<typename T> void f0(T); // expected-note{{here}}
template void f0(int); // expected-error{{explicit instantiation of undefined function template}}

template<typename T>
struct X0 {
  struct Inner;
  
  void f1(); // expected-note{{here}}
  
  static T value; // expected-note{{here}}
};

template void X0<int>::f1(); // expected-error{{explicit instantiation of undefined member function}}

template int X0<int>::value; // expected-error{{explicit instantiation of undefined static data member}}

template<> void f0(long); // expected-note{{previous template specialization is here}}
template void f0(long); // expected-warning{{explicit instantiation of 'f0<long>' that occurs after an explicit specialization has no effect}}

template<> void X0<long>::f1(); // expected-note{{previous template specialization is here}}
template void X0<long>::f1(); // expected-warning{{explicit instantiation of 'f1' that occurs after an explicit specialization has no effect}}

template<> struct X0<long>::Inner; // expected-note{{previous template specialization is here}}
template struct X0<long>::Inner; // expected-warning{{explicit instantiation of 'Inner' that occurs after an explicit specialization has no effect}}

template<> long X0<long>::value; // expected-note{{previous template specialization is here}}
template long X0<long>::value; // expected-warning{{explicit instantiation of 'value' that occurs after an explicit specialization has no effect}}

template<> struct X0<double>; // expected-note{{previous template specialization is here}}
template struct X0<double>; // expected-warning{{explicit instantiation of 'X0<double>' that occurs after an explicit specialization has no effect}}

// PR 6458
namespace test0 {
  template <class T> class foo {
    int compare(T x, T y);
  };

  template <> int foo<char>::compare(char x, char y);
  template <class T> int foo<T>::compare(T x, T y) {
    // invalid at T=char; if we get a diagnostic here, we're
    // inappropriately instantiating this template.
    void *ptr = x;
  }
  extern template class foo<char>; // expected-warning 0-1{{extern templates are a C++11 extension}}
  template class foo<char>;
}
