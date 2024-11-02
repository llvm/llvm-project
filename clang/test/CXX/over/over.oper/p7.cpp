// RUN: %clang_cc1 -std=c++11 %s -verify=expected,cxx11
// RUN: %clang_cc1 -std=c++2b %s -verify=expected,cxx2b
// RUN: %clang_cc1 -std=c++2b -Wpre-c++2b-compat %s -verify=expected,precxx2b


struct Functor {
  static int operator()(int a, int b);
  // cxx11-warning@-1 {{is a C++2b extension}}
  // precxx2b-warning@-2 {{declaring overloaded 'operator()' as 'static' is a C++2b extension}}
};

struct InvalidParsing1 {
  extern int operator()(int a, int b);  // expected-error {{storage class specified}}
};

struct InvalidParsing2 {
  extern static int operator()(int a, int b);  // expected-error {{storage class specified}} // expected-error {{cannot combine with previous 'extern' declaration specifier}}
};
