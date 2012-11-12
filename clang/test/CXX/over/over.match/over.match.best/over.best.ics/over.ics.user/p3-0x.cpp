// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s

namespace PR6285 {
  template<typename T> struct identity 
  { typedef T type; };

  struct D { 
    template<typename T = short> 
    operator typename identity<T>::type(); // expected-note{{candidate}}
  }; 

  int f() { return D(); } // expected-error{{no viable conversion}}
}

