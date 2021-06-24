//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: gcc-10

// unique_ptr

#include <optional>

#include <iterator>

static_assert(!std::indirectly_readable<std::optional<int> >);
static_assert(!std::indirectly_writable<std::optional<int>, int>);
static_assert(!std::weakly_incrementable<std::optional<int> >);
static_assert(!std::indirectly_movable<std::optional<int>, std::optional<int>>);
static_assert(!std::indirectly_movable_storable<std::optional<int>, std::optional<int>>);
