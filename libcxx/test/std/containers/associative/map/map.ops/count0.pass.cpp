//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// XFAIL: c++03, c++11

// <map>

// class map

//       iterator find(const key_type& k);
// const_iterator find(const key_type& k) const;
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
    typedef std::map<int, double, transparent_less> M;
    assert(M().count(C2Int{5}) == 0);
    }
    {
    typedef std::map<int, double, transparent_less_not_referenceable> M;
    assert(M().count(C2Int{5}) == 0);
    }

  return 0;
}
