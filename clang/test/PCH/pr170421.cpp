// RUN: %clang_cc1 -emit-pch -o %t %s
// RUN: %clang_cc1 -include-pch %t -ftime-trace=%t.json -o - %s

// expected-no-diagnostics

#ifndef HEADER_INCLUDED
#define HEADER_INCLUDED

inline namespace {
template<typename T> T g(T v) { return v; }
template<typename T> T f(T v) { return g(v); }
template<typename T> T g();
}

#else

int x;
void i() { f(x); }

#endif
