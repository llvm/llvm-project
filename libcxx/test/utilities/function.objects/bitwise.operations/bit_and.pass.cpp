//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// bit_and

#include <functional>
#include <type_traits>
#include <cassert>

int main()
{
    typedef std::bit_and<int> F;
    const F f = F();
    static_assert((std::is_base_of<std::binary_function<int, int, int>, F>::value), "");
    assert(f(0xEA95, 0xEA95) == 0xEA95);
    assert(f(0xEA95, 0x58D3) == 0x4891);
    assert(f(0x58D3, 0xEA95) == 0x4891);
    assert(f(0x58D3, 0) == 0);
    assert(f(0xFFFF, 0x58D3) == 0x58D3);
}
