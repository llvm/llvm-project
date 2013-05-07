//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// logical_and

#include <functional>
#include <type_traits>
#include <cassert>

int main()
{
    typedef std::logical_and<int> F;
    const F f = F();
    static_assert((std::is_base_of<std::binary_function<int, int, bool>, F>::value), "");
    assert(f(36, 36));
    assert(!f(36, 0));
    assert(!f(0, 36));
    assert(!f(0, 0));
}
