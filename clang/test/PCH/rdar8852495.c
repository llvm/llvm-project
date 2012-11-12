// Test this without pch.
// RUN: %clang_cc1 %s -include %s -verify -fsyntax-only -Wno-sign-compare -Wtautological-compare

// Test with pch.
// RUN: %clang_cc1 %s -emit-pch -o %t -Wsign-compare -Wtautological-compare
// RUN: %clang_cc1 %s -include-pch %t -verify -fsyntax-only -Wno-sign-compare -Wtautological-compare

// expected-no-diagnostics

// This tests that diagnostic mappings from PCH are propagated for #pragma
// diagnostics but not for command-line flags.

#ifndef HEADER
#define HEADER

#pragma clang diagnostic ignored "-Wtautological-compare"

#else

int f() {
  int a;
  int b = a==a;
  unsigned x;
  signed y;
  return x == y;
}

#endif
