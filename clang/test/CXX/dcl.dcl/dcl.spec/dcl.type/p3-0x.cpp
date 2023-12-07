// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s -fcxx-exceptions

using X = struct { // ok
};
template<typename T> using Y = struct { // expected-error {{cannot be defined in a type alias template}}
};

class K {
  virtual ~K();
  operator struct S {} (); // expected-error{{'K::S' cannot be defined in a type specifier}}
};

struct A {};

void f() {
  int arr[3] = {1,2,3};

  for (struct S { S(int) {} } s : arr) { // expected-error {{types may not be defined in a for range declaration}}
  }

  for (struct S { S(int) {} } s : Undeclared); // expected-error{{types may not be defined in a for range declaration}}
                                               // expected-error@-1{{use of undeclared identifier 'Undeclared'}}

  new struct T {}; // expected-error {{allocation of incomplete type 'struct T'}} //expected-note{{forward declaration of 'T'}}
  new struct A {};

  try {} catch (struct U {}) {} // expected-error {{'U' cannot be defined in a type specifier}}

  (void)(struct V { V(int); })0; // expected-error {{'V' cannot be defined in a type specifier}}

  (void)dynamic_cast<struct W {}*>((K*)0); // expected-error {{'W' cannot be defined in a type specifier}}
  (void)static_cast<struct X {}*>(0); // expected-error {{'X' cannot be defined in a type specifier}}
  (void)reinterpret_cast<struct Y {}*>(0); // expected-error {{'Y' cannot be defined in a type specifier}}
  (void)const_cast<struct Z {}*>((const Z*)0); // expected-error {{'Z' cannot be defined in a type specifier}}
}

void g() throw (struct Ex {}) { // expected-error {{'Ex' cannot be defined in a type specifier}}
}

alignas(struct Aa {}) int x; // expected-error {{'Aa' cannot be defined in a type specifier}}

int a = sizeof(struct So {}); // expected-error {{'So' cannot be defined in a type specifier}}
int b = alignof(struct Ao {}); // expected-error {{'Ao' cannot be defined in a type specifier}}

namespace std { struct type_info; }
const std::type_info &ti = typeid(struct Ti {}); // expected-error {{'Ti' cannot be defined in a type specifier}}
