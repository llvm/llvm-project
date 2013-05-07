//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// unary_negate

#include <functional>
#include <type_traits>
#include <cassert>

int main()
{
    typedef std::unary_negate<std::logical_not<int> > F;
    const F f = F(std::logical_not<int>());
    static_assert((std::is_base_of<std::unary_function<int, bool>, F>::value), "");
    assert(f(36));
    assert(!f(0));
}
