//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class X> class auto_ptr;

// template<class Y> operator auto_ptr<Y>() throw();

// REQUIRES: c++03 || c++11 || c++14
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <memory>
#include <cassert>

#include "test_macros.h"
#include "../AB.h"

std::auto_ptr<B>
source()
{
    return std::auto_ptr<B>(new B(1));
}

void
test()
{
    std::auto_ptr<A> ap2(source());
}

int main(int, char**)
{
    test();

  return 0;
}
