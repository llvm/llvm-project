// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1  -std=c++20 %t/foo.cppm -I%t -emit-module-interface -o %t/foo.pcm
// RUN: %clang_cc1  -fprebuilt-module-path=%t -std=c++20 %t/use.cpp -I%t -fsyntax-only -verify

// RUN: %clang_cc1  -std=c++20 %t/foo.cppm -I%t -emit-reduced-module-interface -o %t/foo.pcm
// RUN: %clang_cc1  -fprebuilt-module-path=%t -std=c++20 %t/use.cpp -I%t -fsyntax-only -verify

//--- foo.h
template <typename T>
T u;

template <typename T = int>
T v;

template <int T = 8>
int v2;

template <typename T>
class my_array {};

template <template <typename> typename C = my_array>
int v3;

template <typename T, int *i = nullptr>
T v4;

template <typename T, T *i = nullptr>
T v5;

inline int a = 43;
template <typename T, int *i = &a>
T v6;

inline int b = 43;
template <typename T, T *i = &b>
T v7;

template <int T = (3 > 2)>
int v8;

consteval int getInt() {
  return 55;
}
template <int T = getInt()>
int v9;

//--- foo.cppm
module;
#include "foo.h"
export module foo;


//--- use.cpp
// expected-no-diagnostics
import foo;
#include "foo.h"
