// RUN: %clang_cc1 -std=c++26 -fexperimental-new-constant-interpreter -x c++ %s -verify

struct F {
  bool V;
};
// Nominal cases
// Type
constexpr F f{true};
struct [[=1]] f1 {};
struct [[=1, =F{true}, =f]] f2 {};
struct [[=1]] [[=2]] f3 {};
// Declaration
[[=1]] const F f4{}; // before declarator
const F [[=1]] f40{}; // after declarator
void f41([[=F{false}]]int i) {} // function parameters
template<class T> [[=3]] void f42(T t); // non dep on template decl
// Redeclaration
[[=2, =3, =2]] void f5();
void f5 [[=4, =2]] ();
// Alias
using A1 [[=1]] = int;
// Error case
// Right hand side of a alias declaration
using A2 = [[=2]] int;  // expected-error {{an attribute list cannot appear here}}
using A3 = int [[=2]];  // expected-error {{annotations are not permitted on defining-type-id}}
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
                              // expected-note@#inst-H {{in instantiation of function template specialization 'h<char>' requested here}}
struct T {
  static constexpr int type() { return 0; }
};

void h(int);
void hh() {
  h(0);
  h('0'); // #inst-H
  h(T{});
}

// Handle copying lvalue
struct U {
  constexpr U() = default;
  U(const U&) = delete; // #del-U
};
constexpr U u;
struct [[ =u ]] deletedCopy{}; // expected-error {{annotation requires an eligible copy constructor}}
                      // expected-note@#del-U {{'U' has been explicitly marked deleted here}}

struct [[ =U{} ]] deletedCopy2{}; // expected-error {{annotation requires an eligible copy constructor}}
                      // expected-note@#del-U {{'U' has been explicitly marked deleted here}}

template <class T>
[[=T{}]] void deletedCopy3();

void f_deletedCopy3() {
  deletedCopy3<U>();
}

// Non structural
struct [[="notstructural"]] h3{}; // expected-error {{C++26 annotation attribute requires a value of structural type}}

// Pointer into string literal
struct [[=&"foo"[0]]] h4{}; // expected-error {{C++26 annotation attribute requires an expression usable as a template argument}} \
                               expected-note {{pointer to subobject of string literal is not allowed in a template argument}}
