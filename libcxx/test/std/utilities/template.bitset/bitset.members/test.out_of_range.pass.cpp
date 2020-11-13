//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions

// test constexpr bool test(size_t pos) const;

// Make sure we throw std::out_of_range when calling test() on an OOB index.

#include <bitset>
#include <cassert>
#include <stdexcept>

int main(int, char**) {
    {
        std::bitset<0> v;
        try { v.test(0); assert(false); } catch (std::out_of_range const&) { }
    }
    {
        std::bitset<1> v("0");
        try { v.test(2); assert(false); } catch (std::out_of_range const&) { }
    }
    {
        std::bitset<10> v("0000000000");
        try { v.test(10); assert(false); } catch (std::out_of_range const&) { }
    }

    return 0;
}
