//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// Check that type traits derive from integral_constant

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <cstddef>
#include <type_traits>

#include "test_macros.h"

static_assert(std::is_base_of<std::false_type, std::is_void<int>>::value, "");
static_assert(std::is_base_of<std::true_type, std::is_integral<int>>::value, "");
static_assert(std::is_base_of<std::false_type, std::is_floating_point<int>>::value, "");
static_assert(std::is_base_of<std::false_type, std::is_array<int>>::value, "");
static_assert(std::is_base_of<std::false_type, std::is_enum<int>>::value, "");
static_assert(std::is_base_of<std::false_type, std::is_union<int>>::value, "");
static_assert(std::is_base_of<std::false_type, std::is_class<int>>::value, "");
static_assert(std::is_base_of<std::false_type, std::is_function<int>>::value, "");
static_assert(std::is_base_of<std::false_type, std::is_pointer<int>>::value, "");
static_assert(std::is_base_of<std::false_type, std::is_lvalue_reference<int>>::value, "");
static_assert(std::is_base_of<std::false_type, std::is_rvalue_reference<int>>::value, "");
static_assert(std::is_base_of<std::false_type, std::is_member_object_pointer<int>>::value, "");
static_assert(std::is_base_of<std::false_type, std::is_member_function_pointer<int>>::value, "");
static_assert(std::is_base_of<std::true_type, std::is_fundamental<int>>::value, "");
static_assert(std::is_base_of<std::true_type, std::is_arithmetic<int>>::value, "");
static_assert(std::is_base_of<std::true_type, std::is_scalar<int>>::value, "");
static_assert(std::is_base_of<std::false_type, std::is_object<int&>>::value, "");
static_assert(std::is_base_of<std::false_type, std::is_compound<int>>::value, "");
static_assert(std::is_base_of<std::false_type, std::is_reference<int>>::value, "");
static_assert(std::is_base_of<std::false_type, std::is_member_pointer<int>>::value, "");
static_assert(std::is_base_of<std::false_type, std::is_const<int>>::value, "");
static_assert(std::is_base_of<std::false_type, std::is_volatile<int>>::value, "");
static_assert(std::is_base_of<std::true_type, std::is_trivial<int>>::value, "");
static_assert(std::is_base_of<std::true_type, std::is_trivially_copyable<int>>::value, "");
static_assert(std::is_base_of<std::true_type, std::is_standard_layout<int>>::value, "");
static_assert(std::is_base_of<std::true_type, std::is_pod<int>>::value, "");
static_assert(std::is_base_of<std::false_type, std::is_empty<int>>::value, "");
static_assert(std::is_base_of<std::false_type, std::is_polymorphic<int>>::value, "");
static_assert(std::is_base_of<std::false_type, std::is_abstract<int>>::value, "");
static_assert(std::is_base_of<std::true_type, std::is_signed<int>>::value, "");
static_assert(std::is_base_of<std::false_type, std::is_unsigned<int>>::value, "");
static_assert(std::is_base_of<std::true_type, std::is_constructible<int>>::value, "");
static_assert(std::is_base_of<std::true_type, std::is_trivially_constructible<int>>::value, "");
static_assert(std::is_base_of<std::true_type, std::is_nothrow_constructible<int>>::value, "");
static_assert(std::is_base_of<std::true_type, std::is_default_constructible<int>>::value, "");
static_assert(std::is_base_of<std::true_type, std::is_trivially_default_constructible<int>>::value, "");
static_assert(std::is_base_of<std::true_type, std::is_nothrow_default_constructible<int>>::value, "");
static_assert(std::is_base_of<std::true_type, std::is_copy_constructible<int>>::value, "");
static_assert(std::is_base_of<std::true_type, std::is_trivially_copy_constructible<int>>::value, "");
static_assert(std::is_base_of<std::true_type, std::is_nothrow_copy_constructible<int>>::value, "");
static_assert(std::is_base_of<std::true_type, std::is_move_constructible<int>>::value, "");
static_assert(std::is_base_of<std::true_type, std::is_trivially_move_constructible<int>>::value, "");
static_assert(std::is_base_of<std::true_type, std::is_nothrow_move_constructible<int>>::value, "");
static_assert(std::is_base_of<std::false_type, std::is_assignable<int, int>>::value, "");
static_assert(std::is_base_of<std::false_type, std::is_trivially_assignable<int, int>>::value, "");
static_assert(std::is_base_of<std::false_type, std::is_nothrow_assignable<int, int>>::value, "");
static_assert(std::is_base_of<std::true_type, std::is_copy_assignable<int>>::value, "");
static_assert(std::is_base_of<std::true_type, std::is_trivially_copy_assignable<int>>::value, "");
static_assert(std::is_base_of<std::true_type, std::is_nothrow_copy_assignable<int>>::value, "");
static_assert(std::is_base_of<std::true_type, std::is_move_assignable<int>>::value, "");
static_assert(std::is_base_of<std::true_type, std::is_trivially_move_assignable<int>>::value, "");
static_assert(std::is_base_of<std::true_type, std::is_nothrow_move_assignable<int>>::value, "");
static_assert(std::is_base_of<std::true_type, std::is_destructible<int>>::value, "");
static_assert(std::is_base_of<std::true_type, std::is_trivially_destructible<int>>::value, "");
static_assert(std::is_base_of<std::true_type, std::is_nothrow_destructible<int>>::value, "");
static_assert(std::is_base_of<std::false_type, std::has_virtual_destructor<int>>::value, "");
static_assert(std::is_base_of<std::integral_constant<std::size_t, 1>, std::alignment_of<char>>::value, "");
static_assert(std::is_base_of<std::integral_constant<std::size_t, 0>, std::rank<char>>::value, "");
static_assert(std::is_base_of<std::integral_constant<std::size_t, 0>, std::extent<char>>::value, "");
static_assert(std::is_base_of<std::true_type, std::is_same<int, int>>::value, "");
static_assert(std::is_base_of<std::false_type, std::is_base_of<int, int>>::value, "");
static_assert(std::is_base_of<std::true_type, std::is_convertible<int, int>>::value, "");
#if TEST_STD_VER <= 17
static_assert(std::is_base_of<std::true_type, std::is_literal_type<int>>::value, "");
#endif
#if TEST_STD_VER >= 14
static_assert(std::is_base_of<std::false_type, std::is_null_pointer<int>>::value, "");
static_assert(std::is_base_of<std::false_type, std::is_final<int>>::value, "");
#endif
#if TEST_STD_VER >= 17
static_assert(std::is_base_of<std::true_type, std::has_unique_object_representations<int>>::value, "");
static_assert(std::is_base_of<std::false_type, std::is_aggregate<int>>::value, "");
static_assert(std::is_base_of<std::false_type, std::is_swappable_with<int, int>>::value, "");
static_assert(std::is_base_of<std::true_type, std::is_swappable<int>>::value, "");
static_assert(std::is_base_of<std::false_type, std::is_nothrow_swappable_with<int, int>>::value, "");
static_assert(std::is_base_of<std::true_type, std::is_nothrow_swappable<int>>::value, "");
static_assert(std::is_base_of<std::false_type, std::is_invocable<int>>::value, "");
static_assert(std::is_base_of<std::false_type, std::is_invocable_r<int, int>>::value, "");
static_assert(std::is_base_of<std::false_type, std::is_nothrow_invocable<int, int>>::value, "");
static_assert(std::is_base_of<std::false_type, std::is_nothrow_invocable_r<int, int>>::value, "");
#endif
#if TEST_STD_VER >= 20
static_assert(std::is_base_of<std::false_type, std::is_bounded_array<int>>::value, "");
static_assert(std::is_base_of<std::false_type, std::is_unbounded_array<int>>::value, "");
static_assert(std::is_base_of<std::true_type, std::is_nothrow_convertible<int, int>>::value, "");
#endif
#if TEST_STD_VER >= 23
#  if defined(__cpp_lib_is_implicit_lifetime) && __cpp_lib_is_implicit_lifetime >= 202302L
static_assert(std::is_base_of<std::true_type, std::is_implicit_lifetime<int>>::value, "");
#  endif
static_assert(std::is_base_of<std::false_type, std::is_scoped_enum<int>>::value, "");
#endif
#if TEST_STD_VER >= 26
#  if defined(__cpp_lib_is_virtual_base_of) && __cpp_lib_is_virtual_base_of >= 202406L
static_assert(std::is_base_of<std::false_type, std::is_virtual_base_of<int, int>>::value, "");
#  endif
#endif
