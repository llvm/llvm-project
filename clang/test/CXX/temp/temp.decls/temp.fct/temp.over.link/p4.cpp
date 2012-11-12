// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

// All of these function templates are distinct.
template<typename T> void f0(T) { }
template<typename T, typename U> void f0(T) { }
template<typename T, typename U> void f0(U) { }
void f0();
template<typename T> void f0(T*);
void f0(int);
template<int I> void f0();
template<typename T> void f0();


