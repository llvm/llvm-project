//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03
// UNSUPPORTED: windows
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=0
// UNSUPPORTED: libcxx-no-debug-mode

// <list>

// Call prev(forward_iterator, -1)

#include <iterator>
#include "test_macros.h"
#include "debug_mode_helper.h"

#include "test_iterators.h"

int main(int, char**)
{
    int a[] = {1, 2, 3};

    bidirectional_iterator<int *> bidi(a+1);
	std::prev(bidi, -1);  // should work fine
	std::prev(bidi,  0);  // should work fine
    std::prev(bidi,  1);  // should work fine

    forward_iterator<int *> it(a+1);
    std::prev(it, -1); // should work fine
    std::prev(it,  0); // should work fine
    EXPECT_DEATH( std::prev(it, 1) ); // can't go backwards on a FwdIter

  return 0;
}
