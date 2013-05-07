//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// negate

#include <functional>
#include <type_traits>
#include <cassert>

int main()
{
    typedef std::negate<int> F;
    const F f = F();
    static_assert((std::is_base_of<std::unary_function<int, int>, F>::value), "");
    assert(f(36) == -36);
}
