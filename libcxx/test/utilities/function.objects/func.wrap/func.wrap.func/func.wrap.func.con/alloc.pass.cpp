//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// class function<R(ArgTypes...)>

// template<class A> function(allocator_arg_t, const A&);

#include <functional>
#include <cassert>

#include "../test_allocator.h"

int main()
{
    std::function<int(int)> f(std::allocator_arg, test_allocator<int>());
    assert(!f);
}
