//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: std-at-least-c++26

#include <iterator>
#include <type_traits>
#include <cstddef>
#include <optional>

using iterator = std::optional<int>::iterator;
using const_iterator = std::optional<int>::const_iterator;

// iterator

static_assert(std::random_access_iterator<iterator>);
static_assert(std::contiguous_iterator<iterator>);
static_assert(std::is_same_v<typename std::iterator_traits<iterator>::value_type, int>);
static_assert(std::is_same_v<typename std::iterator_traits<iterator>::difference_type, std::ptrdiff_t>);
static_assert(std::is_same_v<typename std::iterator_traits<iterator>::pointer, int*>);
static_assert(std::is_same_v<typename std::iterator_traits<iterator>::reference, int&>);
static_assert(std::is_same_v<typename std::iterator_traits<iterator>::iterator_category, std::random_access_iterator_tag>);

// const iterator

static_assert(std::random_access_iterator<const_iterator>);
static_assert(std::contiguous_iterator<const_iterator>);
static_assert(std::is_same_v<typename std::iterator_traits<const_iterator>::value_type, int>);
static_assert(std::is_same_v<typename std::iterator_traits<const_iterator>::difference_type, std::ptrdiff_t>);
static_assert(std::is_same_v<typename std::iterator_traits<const_iterator>::pointer, int*>);
static_assert(std::is_same_v<typename std::iterator_traits<const_iterator>::reference, int&>);
static_assert(std::is_same_v<typename std::iterator_traits<const_iterator>::iterator_category, std::random_access_iterator_tag>);
