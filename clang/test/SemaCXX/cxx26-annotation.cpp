// RUN: %clang_cc1 -std=c++26  -x c++ %s -verify

struct F {
  bool V;
};
// Nominal cases
// Type
struct [[=1]] f1 {};
struct [[=1, =F{true}]] f2 {};
struct [[=1]] [[=2]] f3 {};
// Declaration
const [[=1]] F f4{};
// Redeclaration
[[=2, =3, =2]] void f5();
void f5 [[=4, =2]] ();

// Error case
// Mixing annotation and attributes, with or without trailing characters
struct [[nodiscard, =1]] f6 {};  // expected-error {{attribute specifier cannot contain both attributes and annotations}}
struct [[nodiscard, =1,]] f7 {};  // expected-error {{attribute specifier cannot contain both attributes and annotations}}
struct [[=1, nodiscard, ]] f8 {};  // expected-error {{attribute specifier cannot contain both attributes and annotations}}
struct [[=1, nodiscard ]] f9 {};  // expected-error {{attribute specifier cannot contain both attributes and annotations}}
// Mixing attribute using and annotation
struct G {
  [[using CC: =1]] [[=2]] int f;  // expected-error {{annotations are not permitted following an attribute-using-prefix}}
};
// Substituting into an annotation is not in the immediate context
template<class T>
  [[=T::type()]] void h(T t); // expected-error {{type 'char' cannot be used prior to '::' because it has no members}}
                              // expected-note@#inst {{in instantiation of function template specialization 'h<char>' requested here}}
void h(int);
void hh() {
  h(0);
  h('0'); // #inst
}
