// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T, typename U>
struct X1 {
  static const int value = 0;
};

template<typename T, typename U>
struct X1<T*, U*> {
  static const int value = 1;
};

template<typename T>
struct X1<T*, T*> {
  static const int value = 2;
};

template<typename T>
struct X1<const T*, const T*> {
  static const int value = 3;
};

int array0[X1<int, int>::value == 0? 1 : -1];
int array1[X1<int*, float*>::value == 1? 1 : -1];
int array2[X1<int*, int*>::value == 2? 1 : -1];
typedef const int* CIP;
int array3[X1<const int*, CIP>::value == 3? 1 : -1];

template<typename T, typename U>
struct X2 { };

template<typename T, typename U>
struct X2<T*, U> { }; // expected-note{{matches}}

template<typename T, typename U>
struct X2<T, U*> { }; // expected-note{{matches}}

template<typename T, typename U>
struct X2<const T*, const U*> { };

X2<int*, int*> x2a; // expected-error{{ambiguous}}
X2<const int*, const int*> x2b;
