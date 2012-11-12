//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <mutex>

// struct once_flag;

// constexpr once_flag() noexcept;

#include <mutex>

int main()
{
    {
    std::once_flag f;
    }
#ifndef _LIBCPP_HAS_NO_CONSTEXPR
    {
    constexpr std::once_flag f;
    }
#endif
}
