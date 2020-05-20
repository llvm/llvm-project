// RUN: %clang_cc1 -std=c++98 -verify %s

enum E {};
enum F {};

struct A {
  // OK, this is an enumeration bit-field.
  enum E : int(0);
  enum F : int{0}; // expected-error {{expected '(' for function-style cast}}
};
