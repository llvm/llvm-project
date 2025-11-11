// RUN: %clang_cc1 -std=c++26  -x c++ %s -verify

struct F {
  bool V;
};

// Type
struct [[=1]] f1 {};
struct [[=1, =F{true}]] f2 {};
struct [[=1]] [[=2]] f3 {};

// Declaration
const [[=1]] F ff{};

// Redeclaration
[[=2, =3, =2]] void g();
void g [[=4, =2]] ();

// Error case
struct [[nodiscard, =1]] f4 {};  // expected-error {{attribute specifier cannot contain both attributes and annotations}}
struct [[=1, nodiscard, ]] f5 {};  // expected-error {{attribute specifier cannot contain both attributes and annotations}}

struct G {
  [[using CC: =1]] [[=2]] int f;  // expected-error {{annotations are not permitted following an attribute-using-prefix}}
};

template<class T>
  [[=T::type()]] void h(T t); // expected-error {{type 'char' cannot be used prior to '::' because it has no members}}

void h(int);

void hh() {
  h(0);
  h('0'); // expected-note {{in instantiation of function template specialization 'h<char>' requested here}}
}
