//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>
// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17
//
// template <class T>
// class allocator
// {
// public: // All of these are constexpr after C++17
//  constexpr allocator() noexcept;
//  constexpr allocator(const allocator&) noexcept;
//  template<class U> constexpr allocator(const allocator<U>&) noexcept;
// ...
// };

#include <memory>
#include <cstddef>

#include "test_macros.h"


int main()
{
    {
    typedef std::allocator<char> AC;
    typedef std::allocator<long> AL;

    constexpr AC a1;
    constexpr AC a2{a1};
    constexpr AL a3{a2};
    (void) a3;
    }
    {
    typedef std::allocator<const char> AC;
    typedef std::allocator<const long> AL;

    constexpr AC a1;
    constexpr AC a2{a1};
    constexpr AL a3{a2};
    (void) a3;
    }

}
