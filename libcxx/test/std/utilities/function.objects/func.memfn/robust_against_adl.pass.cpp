//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

#include <functional>

#include "test_macros.h"

struct Incomplete;
template<class T> struct Holder { T t; };
typedef Holder<Incomplete> *Ptr;

struct A {
    Ptr no_args() const { return nullptr; }
    Ptr one_arg(Ptr p) const { return p; }
    void one_arg_void(Ptr) const { }
};

int main(int, char**)
{
    A a;
    A *pa = &a;
    const A *cpa = &a;
    Ptr x = nullptr;
    const Ptr cx = nullptr;
    std::mem_fn(&A::no_args)(a);
    std::mem_fn(&A::no_args)(pa);
    std::mem_fn(&A::no_args)(*cpa);
    std::mem_fn(&A::no_args)(cpa);
    std::mem_fn(&A::one_arg)(a, x);
    std::mem_fn(&A::one_arg)(pa, x);
    std::mem_fn(&A::one_arg)(a, cx);
    std::mem_fn(&A::one_arg)(pa, cx);
    std::mem_fn(&A::one_arg)(*cpa, x);
    std::mem_fn(&A::one_arg)(cpa, x);
    std::mem_fn(&A::one_arg)(*cpa, cx);
    std::mem_fn(&A::one_arg)(cpa, cx);
    std::mem_fn(&A::one_arg_void)(a, x);
    std::mem_fn(&A::one_arg_void)(pa, x);
    std::mem_fn(&A::one_arg_void)(a, cx);
    std::mem_fn(&A::one_arg_void)(pa, cx);
    std::mem_fn(&A::one_arg_void)(*cpa, x);
    std::mem_fn(&A::one_arg_void)(cpa, x);
    std::mem_fn(&A::one_arg_void)(*cpa, cx);
    std::mem_fn(&A::one_arg_void)(cpa, cx);
    return 0;
}
