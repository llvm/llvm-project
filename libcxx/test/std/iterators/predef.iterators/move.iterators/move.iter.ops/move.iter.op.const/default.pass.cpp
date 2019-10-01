//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// move_iterator

// move_iterator();
//
//  constexpr in C++17

#include <iterator>

#include "test_macros.h"
#include "test_iterators.h"

template <class It>
void
test()
{
    std::move_iterator<It> r;
    (void)r;
}

int main(int, char**)
{
    test<input_iterator<char*> >();
    test<forward_iterator<char*> >();
    test<bidirectional_iterator<char*> >();
    test<random_access_iterator<char*> >();
    test<char*>();

#if TEST_STD_VER > 14
    {
    constexpr std::move_iterator<const char *> it;
    (void)it;
    }
#endif

  return 0;
}
