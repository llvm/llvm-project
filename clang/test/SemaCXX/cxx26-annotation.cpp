// RUN: %clang_cc1 -std=c++26 -fexperimental-new-constant-interpreter -x c++ %s -verify

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
void f41([[=F{false}]]int i) {} // function parameters
template<class T> [[=3]] void f42(T t); // non dep on template decl
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
  bool V;
  constexpr U(bool v) : V(v) {}
  U(const U&) = delete; // #del-U
};
constexpr U u(true);
struct [[ =u ]] h2{}; // expected-error {{call to deleted constructor of 'U'}}
                      // expected-note@#del-U {{'U' has been explicitly marked deleted here}}

// Non structural
struct [[="notstructural"]] h3{}; // expected-error {{C++26 annotation attribute requires an expression usable as a template argument}} \
                                     expected-note {{reference to string literal is not allowed in a template argument}}

// Pointer into string literal
struct [[=&"foo"[0]]] h4{}; // expected-error {{C++26 annotation attribute requires an expression usable as a template argument}} \
                               expected-note {{pointer to subobject of string literal is not allowed in a template argument}}
