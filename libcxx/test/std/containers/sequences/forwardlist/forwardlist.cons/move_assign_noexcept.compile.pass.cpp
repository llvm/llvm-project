//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <forward_list>

// forward_list& operator=(forward_list&& c)
//     noexcept(
//          allocator_type::propagate_on_container_move_assignment::value &&
//          is_nothrow_move_assignable<allocator_type>::value);

// This tests a conforming extension

// UNSUPPORTED: c++03

#include <forward_list>

#include "test_macros.h"
#include "MoveOnly.h"
#include "test_allocator.h"

template <class T>
struct always_equal_alloc {
  using value_type = T;
  always_equal_alloc(const always_equal_alloc&);
  void allocate(std::size_t);
};

template <class T>
struct not_always_equal_alloc {
  int i;
  using value_type = T;
  not_always_equal_alloc(const not_always_equal_alloc&);
  void allocate(std::size_t);
};

static_assert(std::is_nothrow_move_assignable<std::forward_list<MoveOnly>>::value, "");
static_assert(!std::is_nothrow_move_assignable<std::forward_list<MoveOnly, test_allocator<MoveOnly>>>::value, "");
#if TEST_STD_VER >= 17
static_assert(std::is_nothrow_move_assignable<std::forward_list<MoveOnly, always_equal_alloc<MoveOnly>>>::value, "");
#endif
static_assert(!std::is_nothrow_move_assignable<std::forward_list<MoveOnly, not_always_equal_alloc<MoveOnly>>>::value,
              "");
#if defined(_LIBCPP_VERSION)
static_assert(std::is_nothrow_move_assignable<std::forward_list<MoveOnly, other_allocator<MoveOnly>>>::value, "");
#endif // _LIBCPP_VERSION
