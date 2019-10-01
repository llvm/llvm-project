//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ios>

// template <class charT, class traits> class basic_ios

// operator unspecified-bool-type() const;

#include <ios>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    std::ios ios(0);
    assert(static_cast<bool>(ios) == !ios.fail());
    ios.setstate(std::ios::failbit);
    assert(static_cast<bool>(ios) == !ios.fail());
    static_assert((!std::is_convertible<std::ios, void*>::value), "");
    static_assert((!std::is_convertible<std::ios, int>::value), "");
    static_assert((!std::is_convertible<std::ios const&, int>::value), "");
#if TEST_STD_VER >= 11
    static_assert((!std::is_convertible<std::ios, bool>::value), "");
#endif

  return 0;
}
