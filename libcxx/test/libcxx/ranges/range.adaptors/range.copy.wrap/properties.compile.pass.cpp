//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// Test various properties of <copyable-box>

#include <ranges>

#include <optional>

#include "types.h"

template <class T>
constexpr bool valid_copyable_box = requires {
  typename std::ranges::__copyable_box<T>;
};

struct NotCopyConstructible {
  NotCopyConstructible() = default;
  NotCopyConstructible(NotCopyConstructible&&) = default;
  NotCopyConstructible(NotCopyConstructible const&) = delete;
  NotCopyConstructible& operator=(NotCopyConstructible&&) = default;
  NotCopyConstructible& operator=(NotCopyConstructible const&) = default;
};

static_assert(!valid_copyable_box<void>); // not an object type
static_assert(!valid_copyable_box<int&>); // not an object type
static_assert(!valid_copyable_box<NotCopyConstructible>);

// primary template
static_assert(sizeof(std::ranges::__copyable_box<CopyConstructible>) == sizeof(std::optional<CopyConstructible>));

// optimization #1
static_assert(sizeof(std::ranges::__copyable_box<Copyable>) == sizeof(Copyable));
static_assert(alignof(std::ranges::__copyable_box<Copyable>) == alignof(Copyable));

// optimization #2
static_assert(sizeof(std::ranges::__copyable_box<NothrowCopyConstructible>) == sizeof(NothrowCopyConstructible));
static_assert(alignof(std::ranges::__copyable_box<NothrowCopyConstructible>) == alignof(NothrowCopyConstructible));
