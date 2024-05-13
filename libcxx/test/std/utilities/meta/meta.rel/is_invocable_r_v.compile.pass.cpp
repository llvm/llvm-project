//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// is_invocable_r

#include <type_traits>

// Non-invocable types

static_assert(!std::is_invocable_r_v<void, void>);
static_assert(!std::is_invocable_r_v<void, int>);
static_assert(!std::is_invocable_r_v<void, int*>);
static_assert(!std::is_invocable_r_v<void, int&>);
static_assert(!std::is_invocable_r_v<void, int&&>);

// Result type matches

template <typename T>
T Return();

static_assert(std::is_invocable_r_v<int, decltype(Return<int>)>);
static_assert(std::is_invocable_r_v<char, decltype(Return<char>)>);
static_assert(std::is_invocable_r_v<int*, decltype(Return<int*>)>);
static_assert(std::is_invocable_r_v<int&, decltype(Return<int&>)>);
static_assert(std::is_invocable_r_v<int&&, decltype(Return<int&&>)>);

// void result type

// Any actual return type should be useable with a result type of void.
static_assert(std::is_invocable_r_v<void, decltype(Return<void>)>);
static_assert(std::is_invocable_r_v<void, decltype(Return<int>)>);
static_assert(std::is_invocable_r_v<void, decltype(Return<int*>)>);
static_assert(std::is_invocable_r_v<void, decltype(Return<int&>)>);
static_assert(std::is_invocable_r_v<void, decltype(Return<int&&>)>);

// const- and volatile-qualified void should work too.
static_assert(std::is_invocable_r_v<const void, decltype(Return<void>)>);
static_assert(std::is_invocable_r_v<const void, decltype(Return<int>)>);
static_assert(std::is_invocable_r_v<volatile void, decltype(Return<void>)>);
static_assert(std::is_invocable_r_v<volatile void, decltype(Return<int>)>);
static_assert(std::is_invocable_r_v<const volatile void, decltype(Return<void>)>);
static_assert(std::is_invocable_r_v<const volatile void, decltype(Return<int>)>);

// Conversion of result type

// It should be possible to use a result type to which the actual return type
// can be converted.
static_assert(std::is_invocable_r_v<char, decltype(Return<int>)>);
static_assert(std::is_invocable_r_v<const int*, decltype(Return<int*>)>);
static_assert(std::is_invocable_r_v<void*, decltype(Return<int*>)>);
static_assert(std::is_invocable_r_v<const int&, decltype(Return<int>)>);
static_assert(std::is_invocable_r_v<const int&, decltype(Return<int&>)>);
static_assert(std::is_invocable_r_v<const int&, decltype(Return<int&&>)>);
static_assert(std::is_invocable_r_v<const char&, decltype(Return<int>)>);

// But not a result type where the conversion doesn't work.
static_assert(!std::is_invocable_r_v<int, decltype(Return<void>)>);
static_assert(!std::is_invocable_r_v<int, decltype(Return<int*>)>);

// Non-moveable result type

// Define a type that can't be move-constructed.
struct CantMove {
  CantMove() = default;
  CantMove(CantMove&&) = delete;
};

static_assert(!std::is_move_constructible_v<CantMove>);
static_assert(!std::is_copy_constructible_v<CantMove>);

// Define functions that return that type.
CantMove MakeCantMove() { return {}; }
CantMove MakeCantMoveWithArg(int) { return {}; }

// Assumption check: it should be possible to call one of those functions and
// use it to initialize a CantMove object.
CantMove cant_move = MakeCantMove();

// Therefore std::is_invocable_r should agree that they can be invoked to yield
// a CantMove.
static_assert(std::is_invocable_r_v<CantMove, decltype(MakeCantMove)>);
static_assert(std::is_invocable_r_v<CantMove, decltype(MakeCantMoveWithArg), int>);

// Of course it still shouldn't be possible to call one of the functions and get
// back some other type.
static_assert(!std::is_invocable_r_v<int, decltype(MakeCantMove)>);

// And the argument types should still be important.
static_assert(!std::is_invocable_r_v<CantMove, decltype(MakeCantMove), int>);
static_assert(!std::is_invocable_r_v<CantMove, decltype(MakeCantMoveWithArg)>);

// is_invocable_r

// The struct form should be available too, not just the _v variant.
static_assert(std::is_invocable_r<int, decltype(Return<int>)>::value);
static_assert(!std::is_invocable_r<int*, decltype(Return<int>)>::value);
