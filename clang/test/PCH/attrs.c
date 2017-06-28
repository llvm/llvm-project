// Test this without pch.
// RUN: %clang_cc1 -include %s -fsyntax-only -verify %s

// Test with pch.
// RUN: %clang_cc1 -emit-pch -o %t %s
// RUN: %clang_cc1 -include-pch %t -fsyntax-only -verify %s 

#ifndef HEADER
#define HEADER

int f(int) __attribute__((visibility("default"), overloadable));
int g(int) __attribute__((abi_tag("foo", "bar", "baz"), no_sanitize("address", "memory")));

#else

float f(float);
double f(double); // expected-error{{overloadable}}
                  // expected-note@-2{{previous unmarked overload}}
void h() { g(0); }

#endif
