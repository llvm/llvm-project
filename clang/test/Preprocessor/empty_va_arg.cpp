// RUN: %clang_cc1 -Eonly -std=c++17 -pedantic -verify %s
// RUN: %clang_cc1 -Eonly -std=c17 -pedantic -verify -x c %s
// RUN: %clang_cc1 -Eonly -std=c++20 -pedantic -Wpre-c++20-compat -verify=compat %s

#define FOO(x, ...) // expected-note {{macro 'FOO' defined here}} \
                    // compat-note {{macro 'FOO' defined here}}

int main() {
  FOO(42) // expected-warning {{must specify at least one argument for '...' parameter of variadic macro}} \
          // compat-warning {{passing no argument for the '...' parameter of a variadic macro is incompatible with C++ standards before C++20}}
}

