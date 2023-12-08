//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <set>

// class multiset

// explicit multiset(const key_compare& comp);

#include <set>
#include <cassert>

#include "test_macros.h"
#include "../../../test_compare.h"

int main(int, char**)
{
    typedef test_less<int> C;
    const std::multiset<int, C> m(C(3));
    assert(m.empty());
    assert(m.begin() == m.end());
    assert(m.key_comp() == C(3));
    assert(m.value_comp() == C(3));

  return 0;
}
