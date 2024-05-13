//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_map>

// class unordered_multimap

// allocator_type get_allocator() const

#include <unordered_map>
#include <cassert>
#include <string>

#include "test_allocator.h"
#include "test_macros.h"

int main(int, char**) {
    typedef std::pair<const int, std::string> ValueType;
    {
        std::allocator<ValueType> alloc;
        const std::unordered_multimap<int, std::string> m(alloc);
        assert(m.get_allocator() == alloc);
    }
    {
        other_allocator<ValueType> alloc(1);
        const std::unordered_multimap<int, std::string, std::hash<int>,
                                      std::equal_to<int>,
                                      other_allocator<ValueType> > m(alloc);
        assert(m.get_allocator() == alloc);
    }

    return 0;
}
