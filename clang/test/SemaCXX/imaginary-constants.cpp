// RUN: %clang_cc1 -fsyntax-only -verify %s -include %s -std=gnu++98
// RUN: %clang_cc1 -fsyntax-only -verify %s -include %s -std=c++11
// RUN: %clang_cc1 -fsyntax-only -verify %s -include %s -std=c++14 -DCXX14=1

// expected-no-diagnostics

#ifndef HEADER
#define HEADER

_Complex int val1 = 2i;
_Complex long val2 = 2il;
_Complex long long val3 = 2ill;
_Complex float val4 = 2.0if;
_Complex double val5 = 2.0i;
_Complex long double val6 = 2.0il;

#if CXX14

#pragma clang system_header

namespace std {
  template<typename T> struct complex {};
  complex<float> operator""if(unsigned long long);
  complex<float> operator""if(long double);

  complex<double> operator"" i(unsigned long long);
  complex<double> operator"" i(long double);

  complex<long double> operator"" il(unsigned long long);
  complex<long double> operator"" il(long double);
}

using namespace std;

complex<float> f1 = 2.0if;
complex<float> f2 = 2if;
complex<double> d1 = 2.0i;
complex<double> d2 = 2i;
complex<long double> l1 = 2.0il;
complex<long double> l2 = 2il;

#endif

#endif
