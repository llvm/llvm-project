// RUN: %clang_cc1 -std=c++98 -pedantic-errors -verify=expected,cxx98 %s
// RUN: %clang_cc1 -std=c++11 -pedantic-errors -verify=expected %s
// RUN: %clang_cc1 -std=c++14 -pedantic-errors -verify=expected %s
// RUN: %clang_cc1 -std=c++17 -pedantic-errors -verify=expected %s
// RUN: %clang_cc1 -std=c++20 -pedantic-errors -verify=expected %s
// RUN: %clang_cc1 -std=c++23 -pedantic-errors -verify=expected %s
// RUN: %clang_cc1 -std=c++2c -pedantic-errors -verify=expected %s


namespace cwg3005 { // cwg3005: 21 ready 2025-09-12

void f(
    int _, // #cwg3005-first-param
    int _)
    // expected-error@-1 {{redefinition of parameter '_'}}
    //   expected-note@#cwg3005-first-param {{previous definition is here}}
{
    int _;
    // expected-error@-1 {{redefinition of '_'}}
    // expected-note@#cwg3005-first-param {{previous declaration is here}}
}

namespace cwg3079 { // cwg3079: 22 ready 2025-08-27
  struct A { union {int x;;} u; }; // cxx98-error{{extra ';' inside a union is a C++11 extension}}
  struct B { union {int x;;}; }; // cxx98-error{{extra ';' inside a union is a C++11 extension}}
}

} // namespace cwg3005
