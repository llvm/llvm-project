// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify -Wc++26-extensions %s
// RUN: %clang_cc1 -std=c++2c -fsyntax-only -verify=cxx26 -Wpre-c++26-compat %s

int main() {
  (void) R"abc`@$(foobar)abc`@$";
  //expected-warning@-1 {{'`' in a raw string literal delimiter is a C++2c extension}}
  //expected-warning@-2 {{'@' in a raw string literal delimiter is a C++2c extension}}
  //expected-warning@-3 {{'$' in a raw string literal delimiter is a C++2c extension}}
  //cxx26-warning@-4 {{'`' in a raw string literal delimiter is incompatible with standards before C++2c}}
  //cxx26-warning@-5 {{'@' in a raw string literal delimiter is incompatible with standards before C++2c}}
  //cxx26-warning@-6 {{'$' in a raw string literal delimiter is incompatible with standards before C++2c}}
}
