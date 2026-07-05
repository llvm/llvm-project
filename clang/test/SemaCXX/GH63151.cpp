// RUN: %clang_cc1 -fsyntax-only -verify=expected,narrowing %s
// RUN: %clang_cc1 -fsyntax-only -Wno-c++11-narrowing-const-reference -verify %s

struct A { A(const unsigned &x) {} };

void foo(int p) {
  A a { -1 }; // narrowing-error {{constant expression evaluates to -1 which cannot be narrowed to type 'unsigned int'}}
  A b { 0 };
  A c { p }; // narrowing-error {{non-constant-expression cannot be narrowed from type 'int' to 'unsigned int' in initializer list}}
  A d { 0.5 }; // narrowing-error {{type 'double' cannot be narrowed to 'unsigned int' in initializer list}}
               // expected-warning@-1 {{implicit conversion from 'double' to 'unsigned int' changes value from 0.5 to 0}}
}
