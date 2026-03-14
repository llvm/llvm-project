// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics
template<int i> struct x {
  static const int j = i;
  x<j>* y;
};

template<int i>
const int x<i>::j;

int array0[x<2>::j];

template<typename T>
struct X0 {
  static const unsigned value = sizeof(T);
};

template<typename T>
const unsigned X0<T>::value;

int array1[X0<int>::value == sizeof(int)? 1 : -1];

const unsigned& testX0() { return X0<int>::value; }

int array2[X0<int>::value == sizeof(int)? 1 : -1];

template<typename T>
struct X1 {
  static const unsigned value;
};

template<typename T>
const unsigned X1<T>::value = sizeof(T);

int array3[X1<int>::value == sizeof(int)? 1 : -1];
