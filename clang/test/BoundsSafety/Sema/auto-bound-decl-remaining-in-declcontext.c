
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -Wuninitialized -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fbounds-attributes-objc-experimental -Wuninitialized -verify %s

inline void *a() {
  void *b; // expected-note{{initialize the variable 'b' to silence this warning}}
  return b; // expected-warning{{variable 'b' is uninitialized when used here}}
}
