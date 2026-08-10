// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

// PR8905
template<char C1, char C2>
struct X {
  static const bool value = 0;
};

template<int C1>
struct X<C1, C1> {
  static const bool value = 1;
};

int check0[X<1, 2>::value == 0? 1 : -1];
int check1[X<1, 1>::value == 1? 1 : -1];

template<int, int, int> struct int_values {
  static const unsigned value = 0;
};

template<unsigned char C1, unsigned char C3>
struct int_values<C1, 12, C3> {
  static const unsigned value = 1;
};

int check2[int_values<256, 12, 3>::value == 0? 1 : -1];  
