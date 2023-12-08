// Test this without pch.
// RUN: %clang_cc1 -include %s -std=c++11 -fsyntax-only -verify %s

// Test with pch.
// RUN: %clang_cc1 -x c++-header -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++11 -include-pch %t -fsyntax-only -verify %s 

#ifndef PASS1
#define PASS1
struct foo {
  foo(int) : foo() { }
  foo();
  foo(bool) : foo('c') { }
  foo(char) : foo(true) { }
};
#else
foo::foo() : foo(1) { } // expected-error{{creates a delegation cycle}} \
                        // expected-note{{which delegates to}}

// expected-note@11{{it delegates to}}
// expected-note@13{{it delegates to}}
// expected-error@14{{creates a delegation cycle}}
// expected-note@14{{which delegates to}}
#endif
