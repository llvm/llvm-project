//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>

// class multimap

// size_type count(const key_type& k) const;
//
//   The member function templates find, count, lower_bound, upper_bound, and
// equal_range shall not participate in overload resolution unless the
// qualified-id Compare::is_transparent is valid and denotes a type


#include <map>
#include <cassert>

#include "test_macros.h"
#include "is_transparent.h"

#if TEST_STD_VER <= 11
#error "This test requires is C++14 (or later)"
#else

int main(int, char**)
{
    typedef std::multimap<int, double, transparent_less_no_type> M;

    TEST_IGNORE_NODISCARD M().count(C2Int{5});
}
#endif
