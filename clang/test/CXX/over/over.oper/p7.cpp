// RUN: %clang_cc1 -std=c++11 %s -verify=expected,cxx11
// RUN: %clang_cc1 -std=c++23 %s -verify=expected,cxx23
// RUN: %clang_cc1 -std=c++23 -Wpre-c++23-compat %s -verify=expected,precxx23


struct Functor {
  static int operator()(int a, int b);
  static int operator[](int a1);
  // cxx11-warning@-2 {{declaring overloaded 'operator()' as 'static' is a C++23 extension}}
  // cxx11-warning@-2 {{declaring overloaded 'operator[]' as 'static' is a C++23 extension}}
  // precxx23-warning@-4 {{incompatible with C++ standards before C++23}}
  // precxx23-warning@-4 {{incompatible with C++ standards before C++23}}
};

struct InvalidParsing1 {
  extern int operator()(int a, int b);  // expected-error {{storage class specified}}
  extern int operator[](int a1);  // expected-error {{storage class specified}}
};

struct InvalidParsing2 {
  extern static int operator()(int a, int b);  // expected-error {{storage class specified}} // expected-error {{cannot combine with previous 'extern' declaration specifier}}
  extern static int operator[](int a);  // expected-error {{storage class specified}} // expected-error {{cannot combine with previous 'extern' declaration specifier}}
};
