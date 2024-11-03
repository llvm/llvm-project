//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-filesystem
// UNSUPPORTED: availability-filesystem-missing

// directory_iterator, recursive_directory_iterator

#include "filesystem_include.h"

#include <iterator>

using value_type = fs::directory_entry;

static_assert(std::input_iterator<fs::directory_iterator>);
static_assert(!std::forward_iterator<fs::directory_iterator>);
static_assert(!std::indirectly_writable<fs::directory_iterator, value_type>);
static_assert(!std::incrementable<fs::directory_iterator>);
static_assert(std::sentinel_for<fs::directory_iterator, fs::directory_iterator>);
static_assert(!std::sized_sentinel_for<fs::directory_iterator, fs::directory_iterator>);
static_assert(!std::indirectly_movable<fs::directory_iterator, fs::directory_iterator>);
static_assert(!std::indirectly_movable_storable<fs::directory_iterator, fs::directory_iterator>);
static_assert(!std::indirectly_copyable<fs::directory_iterator, fs::directory_iterator>);
static_assert(!std::indirectly_copyable_storable<fs::directory_iterator, fs::directory_iterator>);
static_assert(!std::indirectly_swappable<fs::directory_iterator, fs::directory_iterator>);

static_assert(std::input_iterator<fs::recursive_directory_iterator>);
static_assert(!std::forward_iterator<fs::recursive_directory_iterator>);
static_assert(!std::indirectly_writable<fs::recursive_directory_iterator, value_type>);
static_assert(!std::incrementable<fs::recursive_directory_iterator>);
static_assert(std::sentinel_for<fs::recursive_directory_iterator, fs::recursive_directory_iterator>);
static_assert(!std::sized_sentinel_for<fs::recursive_directory_iterator, fs::recursive_directory_iterator>);
static_assert(!std::indirectly_movable<fs::recursive_directory_iterator, fs::recursive_directory_iterator>);
static_assert(!std::indirectly_movable_storable<fs::recursive_directory_iterator, fs::recursive_directory_iterator>);
static_assert(!std::indirectly_copyable<fs::recursive_directory_iterator, fs::recursive_directory_iterator>);
static_assert(!std::indirectly_copyable_storable<fs::recursive_directory_iterator, fs::recursive_directory_iterator>);
static_assert(!std::indirectly_swappable<fs::recursive_directory_iterator, fs::recursive_directory_iterator>);
