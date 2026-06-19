// Test this without pch.
// RUN: %clang_cc1 %s -fprofiles -fprofiles-test-profiles -std=c++20 -fsyntax-only -include %s -verify

// Test with pch.
// RUN: %clang_cc1 %s -fprofiles -fprofiles-test-profiles -std=c++20 -emit-pch -o %t
// RUN: %clang_cc1 %s -fprofiles -fprofiles-test-profiles -std=c++20 -fsyntax-only -include-pch %t -verify

#ifndef HEADER
#define HEADER

[[profiles::enforce(test::type_cast)]];

#else

void test() {
  int *p = reinterpret_cast<int*>(0); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
}

#endif
