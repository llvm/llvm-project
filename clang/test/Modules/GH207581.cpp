// RUN: rm -rf %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple -emit-module-interface %t/fmt.cppm -o %t/fmt.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple -emit-module-interface %t/seastar.cppm -o %t/seastar.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple -fmodule-file=fmt=%t/fmt.pcm -fmodule-file=seastar=%t/seastar.pcm -fsyntax-only -verify %t/main.cc

//--- base.h
#ifndef BASE_H
#define BASE_H
namespace demo {
enum { max_packed_args = 6 };
template <bool B, class T, class F> struct cond { using type = F; };
template <class T, class F> struct cond<true, T, F> { using type = T; };
template <class T> struct a_t {};
template <class T> struct b_t {};
template <class Context, int N>
using arg_t = typename cond<(N <= max_packed_args), a_t<Context>, b_t<Context>>::type;
}
#endif

//--- fmt.cppm
module;
#include "base.h"
export module fmt;
export namespace demo { 
    using demo::max_packed_args;
}

//--- seastar.cppm
module;
#include "base.h"
export module seastar;
export namespace seastar { inline int dummy = 0; }

//--- main.cc
// expected-no-diagnostics
import seastar;
import fmt;
#include "base.h"
int main() {
    demo::arg_t<int, 5> a;
    demo::arg_t<int, 7> b;
    return 0;
}
