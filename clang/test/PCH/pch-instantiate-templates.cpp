// Test this without pch, template will be instantiated.
// RUN: %clang_cc1 -fsyntax-only %s -verify=expected

// Test with pch, template will be instantiated in the TU.
// RUN: %clang_cc1 -emit-pch -o %t %s -verify=ok
// RUN: %clang_cc1 -include-pch %t -fsyntax-only %s -verify=expected

// Test with pch with template instantiation in the pch.
// RUN: %clang_cc1 -emit-pch -fpch-instantiate-templates -o %t %s -verify=expected

// ok-no-diagnostics

#ifndef HEADER_H
#define HEADER_H

template <typename T>
struct A {
  T foo() const { return "test"; } // @18
};

double bar(A<double> *a) {
  return a->foo(); // @22
}

#endif

// expected-error@18 {{cannot initialize return object}}
// expected-note@22 {{in instantiation of member function}}
