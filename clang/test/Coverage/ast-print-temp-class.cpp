// RUN: %clang_cc1 -ast-print -std=c++14 %s -v -o %t.1.cpp
// RUN: %clang_cc1 -ast-print -std=c++14 %t.1.cpp -o %t.2.cpp
// RUN: diff %t.1.cpp %t.2.cpp

// Specializations

template<typename T> class C0 {};
template<> class C0<long> {};
template<> class C0<long*> {};
C0<int> c0;

template<int N> class C1 {};
template<> class C1<11> {};
C1<2> c1a;
C1<4> c1b;

template<typename T> class C2a {};
template<typename T> class C2b {};
template<template<typename T> class TC> class C2 {};
template<> class C2<C2a> {};
C2<C2b> c2;


// Default arguments

template<typename T = int> class C10 {};
template<int N = 10> class C11 {};
template<typename T, int N = 22> class C12a {};
//FIXME: template<template<typename T, int N> class TC = C12a> class C12 {};
//FIXME: template<template<typename T> class TC = C12a> class C13 {};


// Partial specializations

template<typename T, typename U> struct C20 {
    T a;
    U b;
};
template<typename T> struct C20<T, int> {
    T a;
};

template<int N, typename U> struct C21 {
    U a;
    U b[N];
};
template<int N> struct C21<N, int> {
    int a[N];
};

template<template<typename T2> class TC, typename U> struct C22 {
    TC<U> a;
    U b;
};
template<template<typename T2> class TC> struct C22<TC, int> {
    TC<int> a;
};


// Declaration only
template<typename T> class C30;
template<> class C30<long>;
template<> class C30<long*>;
extern C30<int> c30;
