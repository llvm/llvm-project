// RUN: %clang_cc1 -fsyntax-only -verify %s

struct S {
  S();  // expected-note {{because type 'S' has a user-declared constructor}}    
};

struct { // expected-error {{anonymous structs and classes must be class members}}
};

struct E {
  struct {
    S x;  // expected-error {{anonymous struct member 'x' has a non-trivial constructor}} 
  };
  static struct {
  };
};
