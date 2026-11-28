// RUN: rm -rf %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/view.cppm -emit-reduced-module-interface -o %t/view.pcm
// RUN: %clang_cc1 -std=c++20 %t/wrapper.cppm -emit-reduced-module-interface \
// RUN:   -fprebuilt-module-path=%t -o %t/wrapper.pcm
// RUN: %clang_cc1 -std=c++20 %t/use.cpp -fprebuilt-module-path=%t -fsyntax-only -verify
//
// RUN: %clang_cc1 -std=c++20 %t/view.cppm -emit-module-interface -o %t/view.pcm
// RUN: %clang_cc1 -std=c++20 %t/wrapper.cppm -emit-module-interface \
// RUN:   -fprebuilt-module-path=%t -o %t/wrapper.pcm
// RUN: %clang_cc1 -std=c++20 %t/use.cpp -fprebuilt-module-path=%t -fsyntax-only -verify

//--- global.h
auto global_local() {
  struct S {};
  return S{};
}

//--- view.cppm
module;
#include "global.h"
export module view;

export using global_local_type = decltype(global_local());

export auto local() {
  struct S {};
  return S{};
}

export namespace associated {
struct Base {};
template <class T> constexpr bool g(T *) { return true; }
}

export auto local_derived() {
  struct S : associated::Base {};
  return S{};
}

//--- wrapper.cppm
export module wrapper;
import view;

export template <class T> struct W {};
export using local_wrapper = W<decltype(local())>;
export using derived_wrapper = W<decltype(local_derived())>;
export using global_wrapper = W<global_local_type>;

//--- use.cpp
// expected-no-diagnostics
import wrapper;

void g(...);

void test(local_wrapper *p, global_wrapper *q) {
  g(p);
  g(q);
}

static_assert(g(static_cast<derived_wrapper *>(nullptr)));
