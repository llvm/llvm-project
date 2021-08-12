//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// shared_ptr

#include <memory>

#include <iterator>

static_assert(std::indirectly_readable<std::shared_ptr<int> >);
static_assert(std::indirectly_writable<std::shared_ptr<int>, int>);
static_assert(!std::weakly_incrementable<std::shared_ptr<int> >);
static_assert(std::indirectly_movable<std::shared_ptr<int>, std::shared_ptr<int>>);
static_assert(std::indirectly_movable_storable<std::shared_ptr<int>, std::shared_ptr<int>>);
static_assert(std::indirectly_swappable<std::shared_ptr<int>, std::shared_ptr<int> >);

static_assert(!std::indirectly_readable<std::shared_ptr<void> >);
static_assert(!std::indirectly_writable<std::shared_ptr<void>, void>);
static_assert(!std::weakly_incrementable<std::shared_ptr<void> >);
static_assert(!std::indirectly_movable<std::shared_ptr<void>, std::shared_ptr<void>>);
static_assert(!std::indirectly_movable_storable<std::shared_ptr<void>, std::shared_ptr<void>>);
