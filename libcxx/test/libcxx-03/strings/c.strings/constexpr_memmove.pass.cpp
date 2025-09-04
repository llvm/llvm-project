//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// _Tp* __constexpr_memmove(_Tp* __dest, _Up* __src, __element_count __n);
//
// General tests for __constexpr_memmove.
//
// In particular, we try to ensure that __constexpr_memmove behaves like
// __builtin_memmove as closely as possible. This means that it produces the
// same effect, but also that it has the same type requirements.
//
// __builtin_memmove only requires that the types are TriviallyCopyable
// (which is interestingly different from both is_trivially_XXX_constructible
// and is_trivially_XXX_assignable), so we use some funky types to test these
// corner cases.

#include <__cxx03/__string/constexpr_c_functions.h>
#include <cassert>
#include <cstdint>
#include <type_traits>

#include "test_macros.h"

// The following types are all TriviallyCopyable, but they are not all
// trivially_{copy,move}_{constructible,assignable}. TriviallyCopyable
// guarantees that the type is *at least* one of the four, but no more
// than that.
struct CopyConstructible {
  CopyConstructible() = default;
  int value           = 0;

  CopyConstructible(const CopyConstructible&)            = default;
  CopyConstructible(CopyConstructible&&)                 = delete;
  CopyConstructible& operator=(const CopyConstructible&) = delete;
  CopyConstructible& operator=(CopyConstructible&&)      = delete;
};

struct MoveConstructible {
  MoveConstructible() = default;
  int value           = 0;

  MoveConstructible(const MoveConstructible&)            = delete;
  MoveConstructible(MoveConstructible&&)                 = default;
  MoveConstructible& operator=(const MoveConstructible&) = delete;
  MoveConstructible& operator=(MoveConstructible&&)      = delete;
};

struct CopyAssignable {
  CopyAssignable() = default;
  int value        = 0;

  CopyAssignable(const CopyAssignable&)            = delete;
  CopyAssignable(CopyAssignable&&)                 = delete;
  CopyAssignable& operator=(const CopyAssignable&) = default;
  CopyAssignable& operator=(CopyAssignable&&)      = delete;
};

struct MoveAssignable {
  MoveAssignable() = default;
  int value        = 0;

  MoveAssignable(const MoveAssignable&)            = delete;
  MoveAssignable(MoveAssignable&&)                 = delete;
  MoveAssignable& operator=(const MoveAssignable&) = delete;
  MoveAssignable& operator=(MoveAssignable&&)      = default;
};

template <class Source, class Dest>
TEST_CONSTEXPR_CXX14 void test_user_defined_types() {
  static_assert(std::is_trivially_copyable<Source>::value, "test the test");
  static_assert(std::is_trivially_copyable<Dest>::value, "test the test");

  // Note that we can't just initialize with an initializer list since some of the types we use here
  // are not copy-constructible, which is required in pre-C++20 Standards for that syntax to work.
  Source src[3];
  src[0].value = 1;
  src[1].value = 2;
  src[2].value = 3;
  Dest dst[3];
  dst[0].value = 111;
  dst[1].value = 111;
  dst[2].value = 111;

  Dest* result = std::__constexpr_memmove(dst, src, std::__element_count(3));
  assert(result == dst);
  assert(dst[0].value == 1);
  assert(dst[1].value == 2);
  assert(dst[2].value == 3);
}

template <class Source, class Dest>
TEST_CONSTEXPR_CXX14 void test_builtin_types() {
  Source src[3] = {1, 2, 3};
  Dest dst[3]   = {111, 111, 111};

  Dest* result = std::__constexpr_memmove(dst, src, std::__element_count(3));
  assert(result == dst);
  assert(dst[0] == 1);
  assert(dst[1] == 2);
  assert(dst[2] == 3);
}

template <class SourcePtr, class DestPtr, class ObjectType>
TEST_CONSTEXPR_CXX14 void test_pointer_types() {
  ObjectType objs[3] = {1, 2, 3};

  SourcePtr src[3] = {objs + 0, objs + 1, objs + 2};
  DestPtr dst[3]   = {nullptr, nullptr, nullptr};

  DestPtr* result = std::__constexpr_memmove(dst, src, std::__element_count(3));
  assert(result == dst);
  assert(dst[0] == objs + 0);
  assert(dst[1] == objs + 1);
  assert(dst[2] == objs + 2);
}

TEST_CONSTEXPR_CXX14 bool test() {
  test_user_defined_types<CopyConstructible, CopyConstructible>();
  test_user_defined_types<MoveConstructible, MoveConstructible>();
  test_user_defined_types<CopyAssignable, CopyAssignable>();
  test_user_defined_types<MoveAssignable, MoveAssignable>();

  test_builtin_types<char, char>();
  test_builtin_types<short, short>();
  test_builtin_types<int, int>();
  test_builtin_types<long, long>();
  test_builtin_types<long long, long long>();

  // Cross-type
  test_builtin_types<std::int16_t, std::uint16_t>();
  test_builtin_types<std::int16_t, char16_t>();
  test_builtin_types<std::int32_t, std::uint32_t>();
  test_builtin_types<std::int32_t, char32_t>();

  test_pointer_types<char*, char*, char>();
  test_pointer_types<int*, int*, int>();
  test_pointer_types<long*, long*, long>();
  test_pointer_types<void*, void*, int>();
  test_pointer_types<int* const, int*, int>();

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 14
  static_assert(test(), "");
#endif
  return 0;
}
