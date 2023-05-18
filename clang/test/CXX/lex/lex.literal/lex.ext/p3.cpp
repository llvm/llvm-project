// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s

int &operator "" _x1 (unsigned long long);
int &i1 = 0x123_x1;

double &operator "" _x1 (const char *);
int &i2 = 45_x1;

template<char...> char &operator "" _x1 ();
int &i3 = 0377_x1;

int &i4 = 90000000000000000000000000000000000000000000000_x1; // expected-error {{integer literal is too large to be represented in any integer type}}

double &operator "" _x2 (const char *);
double &i5 = 123123123123123123123123123123123123123123123_x2;

template<char...Cs> constexpr int operator "" _x3() { return sizeof...(Cs); }
static_assert(123456789012345678901234567890123456789012345678901234567890_x3 == 60, "");
