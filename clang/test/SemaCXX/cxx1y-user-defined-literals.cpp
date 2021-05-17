// RUN: %clang_cc1 -std=c++1y %s -include %s -verify

#ifndef INCLUDED
#define INCLUDED

#pragma clang system_header
namespace std {
  using size_t = decltype(sizeof(0));

  struct duration {};
  duration operator""ns(unsigned long long);
  duration operator""us(unsigned long long);
  duration operator""ms(unsigned long long);
  duration operator""s(unsigned long long);
  duration operator""min(unsigned long long);
  duration operator""h(unsigned long long);

  struct string {};
  string operator""s(const char*, size_t);

  template<typename T> struct complex {};
  complex<float> operator""if(long double);
  complex<float> operator""if(unsigned long long);
  complex<double> operator""i(long double);
  complex<double> operator""i(unsigned long long);
  complex<long double> operator""il(long double);
  complex<long double> operator""il(unsigned long long);
}

#else

using namespace std;
duration a = 1ns, b = 1us, c = 1ms, d = 1s, e = 1min, f = 1h;
string s = "foo"s;
char error = 'x's; // expected-error {{invalid suffix}} expected-error {{expected ';'}}

int _1y = 1y; // expected-error {{invalid suffix}}
int _1b = 1b; // expected-error {{invalid digit}}

complex<float> cf1 = 1if, cf2 = 2.if, cf3 = 0x3if;
complex<double> cd1 = 1i, cd2 = 2.i, cd3 = 0b0110101i;
complex<long double> cld1 = 1il, cld2 = 2.il, cld3 = 0047il;

#endif
