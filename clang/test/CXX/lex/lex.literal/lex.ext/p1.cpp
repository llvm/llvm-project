// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s

void operator "" p31(long double); // expected-warning{{user-defined literal suffixes not starting with '_' are reserved}}
void operator "" _p31(long double);
long double operator "" pi(long double); // expected-warning{{user-defined literal suffixes not starting with '_' are reserved}}

float hexfloat = 0x1p31; // allow hexfloats
