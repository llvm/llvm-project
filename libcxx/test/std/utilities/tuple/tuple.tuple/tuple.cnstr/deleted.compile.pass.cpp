//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <tuple>

// tuple(UTypes&&... u);
// tuple(tuple<UTypes...>& u);
// tuple(const tuple<UTypes...>& u);
// tuple(tuple<UTypes...>&& u);
// tuple(const tuple<UTypes...>&& u);
// tuple(pair<U1, U2>& u);
// tuple(const pair<U1, U2>& u);
// tuple(pair<U1, U2>&& u);
// tuple(const pair<U1, U2>&& u);
//
// The constructors above are defined as deleted if reference_constructs_from_temporary_v
// is true for one of the corresponding tuple elements.

#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

struct X {
  X(int);
};

using Alloc       = std::allocator<int>;
using Tuple       = std::tuple<const X&>;
using PairTuple   = std::tuple<const X&, int>;
using SourceTuple = std::tuple<int>;
using SourcePair  = std::pair<int, int>;

static_assert(std::is_constructible_v<Tuple, X&>);
static_assert(!std::is_constructible_v<Tuple, int>);
static_assert(!std::is_constructible_v<Tuple, std::allocator_arg_t, Alloc, int>);
static_assert(!std::is_constructible_v<Tuple, SourceTuple&>);
static_assert(!std::is_constructible_v<Tuple, std::allocator_arg_t, Alloc, SourceTuple&>);
static_assert(!std::is_constructible_v<Tuple, const SourceTuple&>);
static_assert(!std::is_constructible_v<Tuple, std::allocator_arg_t, Alloc, const SourceTuple&>);
static_assert(!std::is_constructible_v<Tuple, SourceTuple&&>);
static_assert(!std::is_constructible_v<Tuple, std::allocator_arg_t, Alloc, SourceTuple&&>);
static_assert(!std::is_constructible_v<Tuple, const SourceTuple&&>);
static_assert(!std::is_constructible_v<Tuple, std::allocator_arg_t, Alloc, const SourceTuple&&>);
static_assert(!std::is_constructible_v<PairTuple, SourcePair&>);
static_assert(!std::is_constructible_v<PairTuple, std::allocator_arg_t, Alloc, SourcePair&>);
static_assert(!std::is_constructible_v<PairTuple, const SourcePair&>);
static_assert(!std::is_constructible_v<PairTuple, std::allocator_arg_t, Alloc, const SourcePair&>);
static_assert(!std::is_constructible_v<PairTuple, SourcePair&&>);
static_assert(!std::is_constructible_v<PairTuple, std::allocator_arg_t, Alloc, SourcePair&&>);
static_assert(!std::is_constructible_v<PairTuple, const SourcePair&&>);
static_assert(!std::is_constructible_v<PairTuple, std::allocator_arg_t, Alloc, const SourcePair&&>);

int main(int, char**) { return 0; }
