//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// <map>

// class multimap

//       iterator upper_bound(const key_type& k);
// const_iterator upper_bound(const key_type& k) const;
//
//   The member function templates find, count, lower_bound, upper_bound, and
// equal_range shall not participate in overload resolution unless the
// qualified-id Compare::is_transparent is valid and denotes a type


#include <map>
#include <cassert>

#include "test_macros.h"
#include "is_transparent.h"

int main(int, char**)
{
    {
    typedef std::multimap<int, double, transparent_less_no_type> M;

    TEST_IGNORE_NODISCARD M().upper_bound(C2Int{5});
    }
}
