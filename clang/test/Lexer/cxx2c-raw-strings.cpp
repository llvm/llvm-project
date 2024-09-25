// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify=precxx26,expected -Wc++26-extensions %s
// RUN: %clang_cc1 -std=c++2c -fsyntax-only -verify=cxx26,expected -Wpre-c++26-compat %s

int main() {
  (void) R"abc`@$(foobar)abc`@$";
  //precxx26-warning@-1 {{'`' in a raw string literal delimiter is a C++2c extension}}
  //precxx26-warning@-2 {{'@' in a raw string literal delimiter is a C++2c extension}}
  //precxx26-warning@-3 {{'$' in a raw string literal delimiter is a C++2c extension}}
  //cxx26-warning@-4 {{'`' in a raw string literal delimiter is incompatible with standards before C++2c}}
  //cxx26-warning@-5 {{'@' in a raw string literal delimiter is incompatible with standards before C++2c}}
  //cxx26-warning@-6 {{'$' in a raw string literal delimiter is incompatible with standards before C++2c}}

  (void) R"\t()\t";
  // expected-error@-1 {{invalid character '\' in raw string delimiter}}
  // expected-error@-2 {{expected expression}}

  (void) R" () ";
  // expected-error@-1 {{invalid character ' ' in raw string delimiter}}
  // expected-error@-2 {{expected expression}}

  (void) R"\()\";
  // expected-error@-1 {{invalid character '\' in raw string delimiter}}
  // expected-error@-2 {{expected expression}}

  (void) R"@(foo)@";
  // cxx26-warning@-1 {{'@' in a raw string literal delimiter is incompatible with standards before C++2c}}
  // precxx26-warning@-2 {{'@' in a raw string literal delimiter is a C++2c extension}}
}
