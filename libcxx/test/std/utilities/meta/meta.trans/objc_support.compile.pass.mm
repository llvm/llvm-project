//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: objective-c++

// Simple test to check that type traits support Objective-C types.

#include <type_traits>
#include "test_macros.h"

@interface I;
@end

// add_pointer
static_assert(std::is_same<std::add_pointer<id>::type, id*>::value, "");
static_assert(std::is_same<std::add_pointer<I>::type, I*>::value, "");

// add_lvalue_reference
static_assert(std::is_same<std::add_lvalue_reference<id>::type, id&>::value, "");
static_assert(std::is_same<std::add_lvalue_reference<I>::type, I&>::value, "");

// add_rvalue_reference
static_assert(std::is_same<std::add_rvalue_reference<id>::type, id&&>::value, "");
static_assert(std::is_same<std::add_rvalue_reference<I>::type, I&&>::value, "");

// decay
static_assert(std::is_same<std::decay<id>::type, id>::value, "");
static_assert(std::is_same<std::decay<I>::type, I>::value, "");
static_assert(std::is_same<std::decay<id(&)[5]>::type, id*>::value, "");

// __libcpp_is_referenceable
LIBCPP_STATIC_ASSERT(std::__libcpp_is_referenceable<id>::value, "");
LIBCPP_STATIC_ASSERT(std::__libcpp_is_referenceable<id*>::value, "");
LIBCPP_STATIC_ASSERT(std::__libcpp_is_referenceable<id&>::value, "");
LIBCPP_STATIC_ASSERT(std::__libcpp_is_referenceable<id&&>::value, "");
LIBCPP_STATIC_ASSERT(std::__libcpp_is_referenceable<I>::value, "");
LIBCPP_STATIC_ASSERT(std::__libcpp_is_referenceable<I*>::value, "");
LIBCPP_STATIC_ASSERT(std::__libcpp_is_referenceable<I&>::value, "");
LIBCPP_STATIC_ASSERT(std::__libcpp_is_referenceable<I&&>::value, "");

// remove_all_extents
static_assert(std::is_same<std::remove_all_extents<id>::type, id>::value, "");
static_assert(std::is_same<std::remove_all_extents<id[5]>::type, id>::value, "");
static_assert(std::is_same<std::remove_all_extents<id[5][10]>::type, id>::value, "");
static_assert(std::is_same<std::remove_all_extents<I>::type, I>::value, "");

// remove_const
static_assert(std::is_same<std::remove_const<id>::type, id>::value, "");
static_assert(std::is_same<std::remove_const<const id>::type, id>::value, "");
static_assert(std::is_same<std::remove_const<I>::type, I>::value, "");
static_assert(std::is_same<std::remove_const<const I>::type, I>::value, "");

// remove_cv
static_assert(std::is_same<std::remove_cv<id>::type, id>::value, "");
static_assert(std::is_same<std::remove_cv<const volatile id>::type, id>::value, "");
static_assert(std::is_same<std::remove_cv<I>::type, I>::value, "");
static_assert(std::is_same<std::remove_cv<const volatile I>::type, I>::value, "");

#if TEST_STD_VER >= 20
// remove_cvref
static_assert(std::is_same<std::remove_cvref<id>::type, id>::value, "");
static_assert(std::is_same<std::remove_cvref<const volatile id&>::type, id>::value, "");
static_assert(std::is_same<std::remove_cvref<const volatile id&&>::type, id>::value, "");
static_assert(std::is_same<std::remove_cvref<I>::type, I>::value, "");
static_assert(std::is_same<std::remove_cvref<const volatile I&>::type, I>::value, "");
static_assert(std::is_same<std::remove_cvref<const volatile I&&>::type, I>::value, "");
#endif

// remove_extent
static_assert(std::is_same<std::remove_all_extents<id>::type, id>::value, "");
static_assert(std::is_same<std::remove_all_extents<id[5]>::type, id>::value, "");
static_assert(std::is_same<std::remove_all_extents<I>::type, I>::value, "");

// remove_pointer
static_assert(!std::is_same<std::remove_pointer<id>::type, id>::value, "");
// The result of removing and re-adding pointer to `id` should be still `id`.
static_assert(std::is_same<std::remove_pointer<id>::type*, id>::value, "");
static_assert(std::is_same<std::add_pointer<std::remove_pointer<id>::type>::type, id>::value, "");
static_assert(std::is_same<std::remove_pointer<std::add_pointer<id>::type>::type, id>::value, "");

// remove_reference
static_assert(std::is_same<std::remove_reference<id>::type, id>::value, "");
static_assert(std::is_same<std::remove_reference<id&>::type, id>::value, "");
static_assert(std::is_same<std::remove_reference<const id&>::type, const id>::value, "");
static_assert(std::is_same<std::remove_reference<id&&>::type, id>::value, "");
static_assert(std::is_same<std::remove_reference<const id&&>::type, const id>::value, "");
static_assert(std::is_same<std::remove_reference<I>::type, I>::value, "");
static_assert(std::is_same<std::remove_reference<I&>::type, I>::value, "");
static_assert(std::is_same<std::remove_reference<const I&>::type, const I>::value, "");
static_assert(std::is_same<std::remove_reference<I&&>::type, I>::value, "");
static_assert(std::is_same<std::remove_reference<const I&&>::type, const I>::value, "");

// remove_volatile
static_assert(std::is_same<std::remove_volatile<id>::type, id>::value, "");
static_assert(std::is_same<std::remove_volatile<volatile id>::type, id>::value, "");
static_assert(std::is_same<std::remove_volatile<I>::type, I>::value, "");
static_assert(std::is_same<std::remove_volatile<volatile I>::type, I>::value, "");

int main(int, char**) {
  return 0;
}
