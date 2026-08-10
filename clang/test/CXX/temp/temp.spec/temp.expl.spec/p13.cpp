// RUN: %clang_cc1 -fsyntax-only %s

template<typename T> void f(T);

template<> void f(int) { }
void f(int) { }
