//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// <ranges>

// struct view_base { };

#include <ranges>
#include <type_traits>

static_assert(std::is_empty_v<std::ranges::view_base>);
static_assert(std::is_trivial_v<std::ranges::view_base>);

// Make sure we can inherit from it, as it's intended (that wouldn't be the
// case if e.g. it was marked as final).
struct View : std::ranges::view_base { };
