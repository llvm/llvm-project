//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iostream>

// istream cerr;

#include <iostream>
#include <cassert>

int main(int, char**)
{
#if 0
    std::cerr << "Hello World!\n";
#else
#ifdef _LIBCPP_HAS_NO_STDOUT
    assert(std::cerr.tie() == NULL);
#else
    assert(std::cerr.tie() == &std::cout);
#endif
    assert(std::cerr.flags() & std::ios_base::unitbuf);
#endif  // 0

  return 0;
}
