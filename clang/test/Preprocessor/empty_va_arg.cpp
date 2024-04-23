// RUN: %clang_cc1 -Eonly -std=c17 -pedantic -verify=c17,expected -x c %s
// RUN: %clang_cc1 -Eonly -std=c23 -pedantic -Wpre-c23-compat -verify=c23,expected -x c %s
// RUN: %clang_cc1 -Eonly -std=c++17 -pedantic -verify=cxx17,expected %s
// RUN: %clang_cc1 -Eonly -std=c++20 -pedantic -Wpre-c++20-compat -verify=cxx20,expected %s

// silent-no-diagnostics

#define FOO(x, ...) // expected-note  {{macro 'FOO' defined here}}

int main() {
  FOO(42) // c17-warning {{passing no argument for the '...' parameter of a variadic macro is a C23 extension}} \
          // cxx17-warning {{passing no argument for the '...' parameter of a variadic macro is a C++20 extension}} \
          // c23-warning {{passing no argument for the '...' parameter of a variadic macro is incompatible with C standards before C23}} \
          // cxx20-warning {{passing no argument for the '...' parameter of a variadic macro is incompatible with C++ standards before C++20}}
}

