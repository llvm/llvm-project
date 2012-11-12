//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// bit_or

#include <functional>
#include <type_traits>
#include <cassert>

int main()
{
    typedef std::bit_or<int> F;
    const F f = F();
    static_assert((std::is_base_of<std::binary_function<int, int, int>, F>::value), "");
    assert(f(0xEA95, 0xEA95) == 0xEA95);
    assert(f(0xEA95, 0x58D3) == 0xFAD7);
    assert(f(0x58D3, 0xEA95) == 0xFAD7);
    assert(f(0x58D3, 0) == 0x58D3);
    assert(f(0xFFFF, 0x58D3) == 0xFFFF);
}
