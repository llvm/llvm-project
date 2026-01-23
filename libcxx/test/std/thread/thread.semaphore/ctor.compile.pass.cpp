//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <semaphore>

// constexpr explicit counting_semaphore(ptrdiff_t desired);

#include <semaphore>
#include <type_traits>

#include "test_macros.h"

static_assert(!std::is_default_constructible<std::binary_semaphore>::value, "");
static_assert(!std::is_default_constructible<std::counting_semaphore<>>::value, "");

static_assert(!std::is_convertible<int, std::binary_semaphore>::value, "");
static_assert(!std::is_convertible<int, std::counting_semaphore<>>::value, "");

#if TEST_STD_VER > 17
// Test constexpr-constructibility. (But not destructibility.)
constinit std::binary_semaphore bs(1);
constinit std::counting_semaphore<> cs(1);
#endif
