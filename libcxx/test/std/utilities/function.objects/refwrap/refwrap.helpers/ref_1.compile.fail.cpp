//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// reference_wrapper

// template <ObjectType T> reference_wrapper<T> ref(T& t);

// Don't allow binding to a temp

// XFAIL: c++98, c++03

#include <functional>

struct A {};

const A source() {return A();}

int main(int, char**)
{
    std::reference_wrapper<const A> r = std::ref(source());
    (void)r;

    return 0;
}
