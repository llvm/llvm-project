// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s

using size_t = decltype(sizeof(int));

int &operator "" _x1 (const char *);
double &i1 = 'a'_x1; // expected-error {{no matching literal operator}}
double &operator "" _x1 (wchar_t);
double &i2 = L'a'_x1;
double &i3 = 'a'_x1; // expected-error {{no matching literal operator}}
double &i4 = operator"" _x1('a'); // ok

char &operator "" _x1(char16_t);
char &i5 = u'a'_x1; // ok
double &i6 = L'a'_x1; // ok
