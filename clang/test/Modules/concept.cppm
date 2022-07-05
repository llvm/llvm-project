// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/A.cppm -emit-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t -I%t %t/B.cppm -verify

//--- foo.h
#ifndef FOO_H
#define FOO_H

template <class T>
concept Range = requires(T &t) { t.begin(); };

template<class _Tp>
concept __integer_like = true;

template <class _Tp>
concept __member_size = requires(_Tp &&t) { t.size(); };

struct A {
public:
  template <Range T>
  using range_type = T;
};

struct __fn {
  template <__member_size _Tp>
  constexpr __integer_like auto operator()(_Tp&& __t) const {
    return __t.size();
  }
};
#endif

//--- A.cppm
module;
#include "foo.h"
export module A;

//--- B.cppm
// expected-no-diagnostics
module;
#include "foo.h"
export module B;
import A;

void foo() {
    A a;
    struct S {
        int size() { return 0; }
        auto operator+(S s) { return 0; }
    };
    __fn{}(S());
}
