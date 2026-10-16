//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

#include <vector>
#include <type_traits>

#include "test_allocator.h"
#include "test_macros.h"

static_assert(!std::is_default_constructible<std::vector<bool>::reference>::value, "");
static_assert(!std::is_default_constructible<std::vector<bool, test_allocator<bool> >::reference>::value, "");

#if TEST_STD_VER >= 11
void test_no_ambiguity_among_default_constructors(std::enable_if<false>);
void test_no_ambiguity_among_default_constructors(std::vector<bool>::reference);
void test_no_ambiguity_among_default_constructors(std::vector<bool, test_allocator<bool>>::reference);

ASSERT_SAME_TYPE(decltype(test_no_ambiguity_among_default_constructors({})), void);
#endif
