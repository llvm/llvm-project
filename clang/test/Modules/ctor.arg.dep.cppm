// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -std=c++20 %t/A.cppm -I%t -emit-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use.cpp -verify -fsyntax-only
//
//--- foo.h

namespace ns {

struct T {
    T(void*);
};

struct A {
    template <typename F>
    A(F f) : t(&f)  {}

    T t;
};

template <typename T>
void foo(T) {
    auto f = [](){};
    ns::A a(f);
}
}

//--- A.cppm
module;
#include "foo.h"
export module A;
export namespace ns {
    using ns::A;
    using ns::foo;
}

//--- Use.cpp
// expected-no-diagnostics
import A;
void test() {
    ns::foo(5);
}
