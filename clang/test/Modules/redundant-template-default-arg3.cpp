// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1  -std=c++20 %t/foo.cppm -I%t -emit-module-interface -o %t/foo.pcm
// RUN: %clang_cc1  -fprebuilt-module-path=%t -std=c++20 %t/use.cpp -I%t/. -fsyntax-only -verify

// RUN: %clang_cc1  -std=c++20 %t/foo.cppm -I%t -emit-reduced-module-interface -o %t/foo.pcm
// RUN: %clang_cc1  -fprebuilt-module-path=%t -std=c++20 %t/use.cpp -I%t/. -fsyntax-only -verify

//--- foo.h
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

//--- foo_bad.h
template <typename T = double>
T v;

template <int T = 9>
int v2;

template <typename T>
class others_array {};

template <template <typename> typename C = others_array>
int v3;

static int a;
consteval int *getIntPtr() {
  return &a;
}
template <typename T, int *i = getIntPtr()>
T v4;

consteval void *getVoidPtr() {
  return &a;
}
template <typename T, T *i = getVoidPtr()>
T v5;

inline int a_ = 43;
template <typename T, int *i = &a_>
T v6;

inline int b_ = 43;
template <typename T, T *i = &b_>
T v7;

template <int T = -1>
int v8;

consteval int getInt2() {
  return 55;
}
template <int T = getInt2()>
int v9;

//--- foo.cppm
module;
#include "foo.h"
export module foo;

//--- use.cpp
import foo;
#include "foo_bad.h"

// expected-error@foo_bad.h:1 {{template parameter default argument is inconsistent with previous definition}}
// expected-note@foo.h:1 {{previous default template argument defined in module foo.<global>}}
// expected-error@foo_bad.h:4 {{template parameter default argument is inconsistent with previous definition}}
// expected-note@foo.h:4 {{previous default template argument defined in module foo.<global>}}
// expected-error@foo_bad.h:10 {{template parameter default argument is inconsistent with previous definition}}
// expected-note@foo.h:10 {{previous default template argument defined in module foo.<global>}}
// expected-error@foo_bad.h:17 {{template parameter default argument is inconsistent with previous definition}}
// expected-note@foo.h:13 {{previous default template argument defined in module foo.<global>}}
// expected-error@foo_bad.h:23 {{template parameter default argument is inconsistent with previous definition}}
// expected-note@foo.h:16 {{previous default template argument defined in module foo.<global>}}
// expected-error@foo_bad.h:27 {{template parameter default argument is inconsistent with previous definition}}
// expected-note@foo.h:20 {{previous default template argument defined in module foo.<global>}}
// expected-error@foo_bad.h:31 {{template parameter default argument is inconsistent with previous definition}}
// expected-note@foo.h:24 {{previous default template argument defined in module foo.<global>}}
// expected-error@foo_bad.h:34 {{template parameter default argument is inconsistent with previous definition}}
// expected-note@foo.h:27 {{previous default template argument defined in module foo.<global>}}
// expected-error@foo_bad.h:40 {{template parameter default argument is inconsistent with previous definition}}
// expected-note@foo.h:33 {{previous default template argument defined in module foo.<global>}}
