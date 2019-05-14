// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

struct S {
  S();
#if __cplusplus <= 199711L
  // expected-note@-2 {{because type 'S' has a user-provided default constructor}}
#endif
};

struct { // expected-error {{anonymous structs and classes must be class members}} expected-warning {{does not declare anything}}
};

struct E {
  struct {
    S x;
#if __cplusplus <= 199711L
    // expected-error@-2 {{anonymous struct member 'x' has a non-trivial default constructor}}
#endif
  };
  static struct { // expected-warning {{does not declare anything}}
  };
  class {
    int anon_priv_field; // expected-error {{anonymous struct cannot contain a private data member}}
  };
};

template <class T> void foo(T);
typedef struct { // expected-note {{use a tag name here to establish linkage prior to definition}}
#if __cplusplus <= 199711L
// expected-note@-2 {{declared here}}
#endif

  void test() {
    foo(this);
#if __cplusplus <= 199711L
    // expected-warning@-2 {{template argument uses unnamed type}}
#endif
  }
} A; // expected-error {{unsupported: typedef changes linkage of anonymous type, but linkage was already computed}}
