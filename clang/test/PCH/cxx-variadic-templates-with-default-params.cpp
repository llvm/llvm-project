// Test this without pch.
// RUN: %clang_cc1 -std=c++11 -include %s -fsyntax-only -verify %s

// Test with pch.
// RUN: %clang_cc1 -std=c++11 -x c++-header -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++11 -include-pch %t -fsyntax-only -verify %s

// expected-no-diagnostics

// PR25271: Ensure that default template arguments prior to a parameter pack
// successfully round-trip.
#ifndef HEADER
#define HEADER
template<unsigned T=123, unsigned... U>
class dummy;

template<unsigned T, unsigned... U>
class dummy {
    int field[T];
};
#else
void f() {
    dummy<> x;
    (void)x;
}
#endif
