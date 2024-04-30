// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -std=c++20 %t/A.cppm -I%t -emit-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use.cpp -verify -fsyntax-only
//
// RUN: %clang_cc1 -std=c++20 %t/A.cppm -I%t -emit-reduced-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use.cpp -verify -fsyntax-only
//
//--- foo.h

namespace ns {
struct A {};

template<typename Func>
constexpr bool __call_is_nt(A)
{
    return true;
}
ns::A make();

template <typename T>
bool foo(T t) {
    auto func = [](){};
    return __call_is_nt<decltype(func)>(t);
}
}

//--- A.cppm
module;
#include "foo.h"
export module A;
export namespace ns {
    using ns::foo;
    using ns::make;
}

//--- Use.cpp
// expected-no-diagnostics
import A;
void test() {
    auto a = ns::make();
    ns::foo(a);
}
