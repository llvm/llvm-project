// RUN: %clang_cc1 -std=c++20 -fexperimental-cxx-profiles \
// RUN:   -fcxx-profiles-preview=std::casting -fsyntax-only -verify %s

// In preview mode, violations are warnings.

char *f(int *p) {
  return reinterpret_cast<char *>(p);
  // expected-warning@-1 {{std::casting forbids reinterpret_cast from 'int *' to 'char *'}}
}
