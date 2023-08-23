// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s

int &operator "" _x1 (long double);
int &i1 = 0.123_x1;

double &operator "" _x1 (const char *);
int &i2 = 45._x1;

template<char...> char &operator "" _x1 ();
int &i3 = 0377e-1_x1;

int &i4 = 1e1000000_x1; // expected-warning {{too large for type 'long double'}}

double &operator "" _x2 (const char *);
double &i5 = 1e1000000_x2;

template<char...Cs> constexpr int operator "" _x3() { return sizeof...(Cs); }
static_assert(1e1000000_x3 == 9, "");
