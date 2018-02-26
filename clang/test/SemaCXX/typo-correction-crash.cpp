// RUN: %clang_cc1 -fsyntax-only -std=c++17 -verify %s
auto check1() {
  return 1;
  return s; // expected-error {{use of undeclared identifier 's'}}
}

int test = 11; // expected-note {{'test' declared here}}
auto check2() {
  return "s";
  return tes; // expected-error {{use of undeclared identifier 'tes'; did you mean 'test'?}}
}

namespace BarNamespace {
namespace NestedNamespace { // expected-note {{'BarNamespace::NestedNamespace' declared here}}
typedef int type;
}
}
struct FooRecord { };
FooRecord::NestedNamespace::type x; // expected-error {{no member named 'NestedNamespace' in 'FooRecord'; did you mean 'BarNamespace::NestedNamespace'?}}

void cast_expr(int g) { +int(n)(g); } // expected-error {{undeclared identifier 'n'}}

void bind() { for (const auto& [test,_] : _test_) { }; } // expected-error {{undeclared identifier '_test_'}}
