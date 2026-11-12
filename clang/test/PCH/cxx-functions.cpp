// Test this without pch.
// RUN: %clang_cc1 -include %S/cxx-functions.h -fsyntax-only -verify -Wno-dynamic-exception-spec %std_cxx98- %s

// RUN: %clang_cc1 -x c++-header -Wno-dynamic-exception-spec -emit-pch -o %t %S/cxx-functions.h
// RUN: %clang_cc1 -include-pch %t -fsyntax-only -verify -Wno-dynamic-exception-spec %s

// expected-no-diagnostics


void test_foo() {
  foo();
}
