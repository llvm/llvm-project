// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T>
void test1(T __attribute__((noescape)) arr, int size);

// expected-warning@+1 {{'noescape' attribute only applies to pointer arguments}}
void test2(int __attribute__((noescape)) arr, int size);