template<typename T>
struct X0;

template<int I>
struct X1;

template<int I>
struct X2;

template<int I>
struct X3;

template<template<int I> class>
struct X4;

template<template<long> class>
struct X5;

template<typename>
struct X6;

extern X0<int> *x0i;
extern X0<long> *x0l;
extern X0<float> *x0r;

template<>
struct X0<char> {
  int member;
};

template<>
struct X0<wchar_t> {
  int member;
};
