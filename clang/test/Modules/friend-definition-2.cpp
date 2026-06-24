// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++14 -x c++ -fmodules -fmodule-name=A -emit-module %t/a.modulemap -o %t/a.pcm
// RUN: %clang_cc1 -std=c++14 -x c++ -fmodules -fmodule-name=B -emit-module %t/b.modulemap -o %t/b.pcm
// RUN: %clang_cc1 -std=c++14 -x c++ -fmodules -fmodule-map-file=%t/a.modulemap -fmodule-map-file=%t/b.modulemap \
// RUN:   -fmodule-file=%t/a.pcm -fmodule-file=%t/b.pcm \
// RUN:   %t/use.cc -verify

// RUN: rm -f %t/*.pcm

// RUN: %clang_cc1 -std=c++14 -x c++ -fmodules -fmodule-name=A -emit-module %t/a.modulemap -o %t/a.pcm -triple i686-windows
// RUN: %clang_cc1 -std=c++14 -x c++ -fmodules -fmodule-name=B -emit-module %t/b.modulemap -o %t/b.pcm -triple i686-windows
// RUN: %clang_cc1 -std=c++14 -x c++ -fmodules -fmodule-map-file=%t/a.modulemap -fmodule-map-file=%t/b.modulemap \
// RUN:   -fmodule-file=%t/a.pcm -fmodule-file=%t/b.pcm \
// RUN:   %t/use.cc -verify -triple i686-windows

//--- a.modulemap
module A {
  header "a.h"
}

//--- a.h
#ifndef A_H
#define A_H
template<typename T> struct ct { friend auto operator-(ct, ct) { struct X {}; return X(); } void x(); };
#endif

//--- b.modulemap
module B {
  header "b.h"
}

//--- b.h
#ifndef B_H
#define B_H
template<typename T> struct ct { friend auto operator-(ct, ct) { struct X {}; return X(); } void x(); };
inline auto f() { return ct<float>() - ct<float>(); }
#endif

//--- use.cc
// expected-no-diagnostics
// Force the definition of ct in module A to be the primary definition.
#include "a.h"
template<typename T> void ct<T>::x() {}

// Attempt to cause the definition of operator- in the ct primary template in
// module B to be the primary definition of that function. If that happens,
// we'll be left with a class template ct that appears to not contain a
// definition of the inline friend function.
#include "b.h"
auto v = f();

ct<int> make();
void h() {
  make() - make();
}
