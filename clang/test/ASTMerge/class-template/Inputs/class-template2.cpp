template<class T>
struct X0;

template<int I>
struct X1;

template<long I>
struct X2;

template<typename>
struct X3;

template<template<int I> class>
struct X4;

template<template<int I> class>
struct X5;

template<template<int I> class>
struct X6;

typedef int Integer;
extern X0<Integer> *x0i;
extern X0<float> *x0f;
extern X0<double> *x0r;

template<>
struct X0<char> {
  int member;
};

template<>
struct X0<wchar_t> {
  float member;
};
