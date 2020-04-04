//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

#include <cstddef>
#include <type_traits>

// max_align_t is a trivial standard-layout type whose alignment requirement
//   is at least as great as that of every scalar type

#include <stdio.h>
#include "test_macros.h"

int main(int, char**)
{

#if TEST_STD_VER > 17
//  P0767
    static_assert(std::is_trivial<std::max_align_t>::value,
                  "std::is_trivial<std::max_align_t>::value");
    static_assert(std::is_standard_layout<std::max_align_t>::value,
                  "std::is_standard_layout<std::max_align_t>::value");
#else
    static_assert(std::is_pod<std::max_align_t>::value,
                  "std::is_pod<std::max_align_t>::value");
#endif
    static_assert((std::alignment_of<std::max_align_t>::value >=
                  std::alignment_of<long long>::value),
                  "std::alignment_of<std::max_align_t>::value >= "
                  "std::alignment_of<long long>::value");
    static_assert(std::alignment_of<std::max_align_t>::value >=
                  std::alignment_of<long double>::value,
                  "std::alignment_of<std::max_align_t>::value >= "
                  "std::alignment_of<long double>::value");
    static_assert(std::alignment_of<std::max_align_t>::value >=
                  std::alignment_of<void*>::value,
                  "std::alignment_of<std::max_align_t>::value >= "
                  "std::alignment_of<void*>::value");

#ifdef __STDCPP_DEFAULT_NEW_ALIGNMENT__
    static_assert(std::alignment_of<std::max_align_t>::value <=
                  __STDCPP_DEFAULT_NEW_ALIGNMENT__,
                  "max_align_t alignment is no larger than new alignment");
#endif

  return 0;
}
