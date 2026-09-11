//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

#include <inplace_vector>
#include <type_traits>

struct Trivial {
  int value;
};

struct NonTrivialDtor {
  NonTrivialDtor()                                 = default;
  NonTrivialDtor(const NonTrivialDtor&)            = default;
  NonTrivialDtor(NonTrivialDtor&&)                 = default;
  NonTrivialDtor& operator=(const NonTrivialDtor&) = default;
  NonTrivialDtor& operator=(NonTrivialDtor&&)      = default;
  ~NonTrivialDtor() {}
};

struct NonTrivialCopyCtor {
  NonTrivialCopyCtor() = default;
  NonTrivialCopyCtor(const NonTrivialCopyCtor&) {}
  NonTrivialCopyCtor(NonTrivialCopyCtor&&)                 = default;
  NonTrivialCopyCtor& operator=(const NonTrivialCopyCtor&) = default;
  NonTrivialCopyCtor& operator=(NonTrivialCopyCtor&&)      = default;
  ~NonTrivialCopyCtor()                                    = default;
};

struct NonTrivialMoveCtor {
  NonTrivialMoveCtor()                          = default;
  NonTrivialMoveCtor(const NonTrivialMoveCtor&) = default;
  NonTrivialMoveCtor(NonTrivialMoveCtor&&) {}
  NonTrivialMoveCtor& operator=(const NonTrivialMoveCtor&) = default;
  NonTrivialMoveCtor& operator=(NonTrivialMoveCtor&&)      = default;
  ~NonTrivialMoveCtor()                                    = default;
};

struct NonTrivialCopyAssign {
  NonTrivialCopyAssign()                            = default;
  NonTrivialCopyAssign(const NonTrivialCopyAssign&) = default;
  NonTrivialCopyAssign(NonTrivialCopyAssign&&)      = default;
  NonTrivialCopyAssign& operator=(const NonTrivialCopyAssign&) { return *this; }
  NonTrivialCopyAssign& operator=(NonTrivialCopyAssign&&) = default;
  ~NonTrivialCopyAssign()                                 = default;
};

struct NonTrivialMoveAssign {
  NonTrivialMoveAssign()                                       = default;
  NonTrivialMoveAssign(const NonTrivialMoveAssign&)            = default;
  NonTrivialMoveAssign(NonTrivialMoveAssign&&)                 = default;
  NonTrivialMoveAssign& operator=(const NonTrivialMoveAssign&) = default;
  NonTrivialMoveAssign& operator=(NonTrivialMoveAssign&&) { return *this; }
  ~NonTrivialMoveAssign() = default;
};

struct MoveOnlyTrivial {
  MoveOnlyTrivial()                                  = default;
  MoveOnlyTrivial(const MoveOnlyTrivial&)            = delete;
  MoveOnlyTrivial(MoveOnlyTrivial&&)                 = default;
  MoveOnlyTrivial& operator=(const MoveOnlyTrivial&) = delete;
  MoveOnlyTrivial& operator=(MoveOnlyTrivial&&)      = default;
  ~MoveOnlyTrivial()                                 = default;
};

struct NonTrivialAll {
  NonTrivialAll() {}
  NonTrivialAll(const NonTrivialAll&) {}
  NonTrivialAll(NonTrivialAll&&) {}
  NonTrivialAll& operator=(const NonTrivialAll&) { return *this; }
  NonTrivialAll& operator=(NonTrivialAll&&) { return *this; }
  ~NonTrivialAll() {}
};

struct NoDefaultCtor {
  explicit NoDefaultCtor(int);
};

template <class T, unsigned long long _Size>
constexpr bool check() {
  using V = std::inplace_vector<T, _Size>;

  static_assert(std::is_trivially_copy_constructible_v<V> == std::is_trivially_copy_constructible_v<T>);
  static_assert(std::is_trivially_move_constructible_v<V> == std::is_trivially_move_constructible_v<T>);
  static_assert(std::is_trivially_destructible_v<V> == std::is_trivially_destructible_v<T>);
  static_assert(std::is_trivially_copy_assignable_v<V> ==
                (std::is_trivially_destructible_v<T> && std::is_trivially_copy_constructible_v<T> &&
                 std::is_trivially_copy_assignable_v<T>));
  static_assert(std::is_trivially_move_assignable_v<V> ==
                (std::is_trivially_destructible_v<T> && std::is_trivially_move_constructible_v<T> &&
                 std::is_trivially_move_assignable_v<T>));

  static_assert(!std::is_trivially_default_constructible_v<V>);

  return true;
}

// N == 0 always trivial
template <class T>
constexpr bool zero_capacity() {
  using V = std::inplace_vector<T, 0>;

  static_assert(std::is_trivially_copyable_v<V>);
  static_assert(std::is_trivially_default_constructible_v<V>);
  static_assert(std::is_trivially_copy_constructible_v<V>);
  static_assert(std::is_trivially_move_constructible_v<V>);
  static_assert(std::is_trivially_copy_assignable_v<V>);
  static_assert(std::is_trivially_move_assignable_v<V>);
  static_assert(std::is_trivially_destructible_v<V>);

  return true;
}

static_assert(check<int, 4>());
static_assert(check<Trivial, 4>());
static_assert(check<NonTrivialDtor, 4>());
static_assert(check<NonTrivialCopyCtor, 4>());
static_assert(check<NonTrivialMoveCtor, 4>());
static_assert(check<NonTrivialCopyAssign, 4>());
static_assert(check<NonTrivialMoveAssign, 4>());
static_assert(check<MoveOnlyTrivial, 4>());
static_assert(check<NonTrivialAll, 4>());

static_assert(zero_capacity<int>());
static_assert(zero_capacity<Trivial>());
static_assert(zero_capacity<NonTrivialDtor>());
static_assert(zero_capacity<NonTrivialCopyCtor>());
static_assert(zero_capacity<NonTrivialMoveCtor>());
static_assert(zero_capacity<NonTrivialCopyAssign>());
static_assert(zero_capacity<NonTrivialMoveAssign>());
static_assert(zero_capacity<MoveOnlyTrivial>());
static_assert(zero_capacity<NonTrivialAll>());
static_assert(zero_capacity<NoDefaultCtor>());
