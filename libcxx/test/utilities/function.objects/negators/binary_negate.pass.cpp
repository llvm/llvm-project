//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// binary_negate

#include <functional>
#include <type_traits>
#include <cassert>

int main()
{
    typedef std::binary_negate<std::logical_and<int> > F;
    const F f = F(std::logical_and<int>());
    static_assert((std::is_base_of<std::binary_function<int, int, bool>, F>::value), "");
    assert(!f(36, 36));
    assert( f(36, 0));
    assert( f(0, 36));
    assert( f(0, 0));
}
