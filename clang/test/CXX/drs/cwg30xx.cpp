// RUN: %clang_cc1 -std=c++98 -pedantic-errors -verify=expected,cxx98 %s
// RUN: %clang_cc1 -std=c++11 -pedantic-errors -verify=expected,since-cxx11,cxx11-23 %s
// RUN: %clang_cc1 -std=c++14 -pedantic-errors -verify=expected,since-cxx11,cxx11-23 %s
// RUN: %clang_cc1 -std=c++17 -pedantic-errors -verify=expected,since-cxx11,cxx11-23 %s
// RUN: %clang_cc1 -std=c++20 -pedantic-errors -verify=expected,since-cxx11,cxx11-23,since-cxx20 %s
// RUN: %clang_cc1 -std=c++23 -pedantic-errors -verify=expected,since-cxx11,cxx11-23,since-cxx20,since-cxx23 %s
// RUN: %clang_cc1 -std=c++2c -pedantic-errors -verify=expected,since-cxx11,since-cxx20,since-cxx23,since-cxx26 %s


namespace cwg3005 { // cwg3005: 21 open 2025-03-10

void f(int _, // expected-note  {{previous declaration is here}} \
              // expected-note  {{previous definition is here}}
       int _) // expected-error {{redefinition of parameter '_'}}
{
    int _; // expected-error {{redefinition of '_'}}
}

}
