//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

#include <cstddef>
#include <type_traits>

// max_align_t is a trivial standard-layout type whose alignment requirement
//   is at least as great as that of every scalar type

#include "test_macros.h"

static_assert(std::is_trivial<std::max_align_t>::value, "");
static_assert(std::is_standard_layout<std::max_align_t>::value, "");
#if TEST_STD_VER <= 17
static_assert(std::is_pod<std::max_align_t>::value, "");
#endif
static_assert(alignof(std::max_align_t) >= alignof(long long), "");
static_assert(alignof(std::max_align_t) >= alignof(long double), "");
static_assert(alignof(std::max_align_t) >= alignof(void*), "");
#if TEST_STD_VER > 14
static_assert(alignof(std::max_align_t) <= __STDCPP_DEFAULT_NEW_ALIGNMENT__,
              "std::max_align_t alignment should be no larger than operator new's alignment");
#endif
