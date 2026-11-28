//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// UNSUPPORTED: libcpp-has-no-experimental-hive

// <hive>

// libc++'s hard and default limits: a block holds at most 4096 elements, and no more bytes than the allocator's
// size_type can count. By default it holds at most 128 KB of elements, but never fewer than 64 elements.

#include <cstddef>
#include <cstdint>
#include <hive>
#include <limits>
#include <type_traits>

#include "sized_allocator.h"

struct Big {
  char data[1000];
};

static_assert(std::hive<char>::block_capacity_hard_limits().min == 1);
static_assert(std::hive<char>::block_capacity_hard_limits().max == 4096);
static_assert(std::hive<int>::block_capacity_hard_limits().max == 4096);
static_assert(std::hive<Big>::block_capacity_hard_limits().max == 4096);

// SIZE_MAX is a multiple of 17, so 17 elements fill size_t exactly and leave no room for the words.
static_assert(std::numeric_limits<std::size_t>::max() % 17 == 0);
struct Huge {
  char data[std::numeric_limits<std::size_t>::max() / 17];
};

static_assert(std::hive<Huge>::block_capacity_hard_limits().max == 16);
static_assert(std::hive<Huge>::block_capacity_default_limits().min == 16);
static_assert(std::hive<Huge>::block_capacity_default_limits().max == 16);

struct alignas(64) Line {
  char data[64];
};

static_assert(std::hive<char>::block_capacity_default_limits().min == 64);
static_assert(std::hive<char>::block_capacity_default_limits().max == 4096);
static_assert(std::hive<long long>::block_capacity_default_limits().max == 4096);
static_assert(std::hive<Line>::block_capacity_default_limits().max == 2048);
static_assert(std::hive<Big>::block_capacity_default_limits().min == 64);
static_assert(std::hive<Big>::block_capacity_default_limits().max == 131);

// For elements above 2 KiB the default minimum of 64 elements decides, and a block exceeds 128 KB.
struct Page {
  char data[4096];
};

static_assert(std::hive<Page>::block_capacity_default_limits().min == 64);
static_assert(std::hive<Page>::block_capacity_default_limits().max == 64);

// An 8-bit size_type counts 255 bytes, which is 31 units of 8 bytes when the words are 8-byte aligned: one word and
// 60 ints, or four words and 216 chars. A 16-bit size_type counts 65 elements of 1000 bytes.
template <class T, class SizeType>
using NarrowHive = std::hive<T, sized_allocator<T, SizeType, std::make_signed_t<SizeType>>>;

static_assert(alignof(std::uint64_t) != 8 || NarrowHive<int, std::uint8_t>::block_capacity_hard_limits().max == 60);
static_assert(alignof(std::uint64_t) != 8 || NarrowHive<char, std::uint8_t>::block_capacity_hard_limits().max == 216);
static_assert(alignof(std::uint64_t) != 8 || NarrowHive<int, std::uint8_t>::block_capacity_default_limits().min == 60);
static_assert(alignof(std::uint64_t) != 8 || NarrowHive<int, std::uint8_t>::block_capacity_default_limits().max == 60);
static_assert(NarrowHive<Big, std::uint16_t>::block_capacity_hard_limits().max == 65);
static_assert(NarrowHive<int, std::uint16_t>::block_capacity_hard_limits().max == 4096);
