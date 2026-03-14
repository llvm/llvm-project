// Test this without pch.
// RUN: %clang_cc1 -fsyntax-only -verify -DBODY %s

// Test with pch.
// RUN: %clang_cc1 -emit-pch -o %t %s
// RUN: %clang_cc1 -include-pch %t -fsyntax-only -verify -DBODY %s

// RUN: %clang_cc1 -emit-pch -fpch-instantiate-templates -o %t %s
// RUN: %clang_cc1 -include-pch %t -fsyntax-only -verify -DBODY %s

#ifndef HEADER_H
#define HEADER_H

template <typename T>
struct A {
  int foo() const;
};

int bar(A<double> *a) {
  return a->foo();
}

#endif // HEADER_H

#ifdef BODY

template <>
int A<double>::foo() const { // expected-error {{explicit specialization of 'foo' after instantiation}}  // expected-note@20 {{implicit instantiation first required here}}
  return 10;
}

#endif // BODY
