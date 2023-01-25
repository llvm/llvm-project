// RUN: %clang_cc1 -std=c++11 %s -verify=expected,cxx11
// RUN: %clang_cc1 -std=c++2b %s -verify=expected,cxx2b
// RUN: %clang_cc1 -std=c++2b -Wpre-c++2b-compat %s -verify=expected,precxx2b


struct Functor {
  static int operator()(int a, int b);
  static int operator[](int a1);
  // cxx11-warning@-2 {{declaring overloaded 'operator()' as 'static' is a C++2b extension}}
  // cxx11-warning@-2 {{declaring overloaded 'operator[]' as 'static' is a C++2b extension}}
  // precxx2b-warning@-4 {{incompatible with C++ standards before C++2b}}
  // precxx2b-warning@-4 {{incompatible with C++ standards before C++2b}}
};

struct InvalidParsing1 {
  extern int operator()(int a, int b);  // expected-error {{storage class specified}}
  extern int operator[](int a1);  // expected-error {{storage class specified}}
};

struct InvalidParsing2 {
  extern static int operator()(int a, int b);  // expected-error {{storage class specified}} // expected-error {{cannot combine with previous 'extern' declaration specifier}}
  extern static int operator[](int a);  // expected-error {{storage class specified}} // expected-error {{cannot combine with previous 'extern' declaration specifier}}
};
