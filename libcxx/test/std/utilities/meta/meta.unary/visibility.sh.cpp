//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// Make sure that the types and variables have the correct visibility attributes

// RUN: %{cxx} %s %{flags} %{compile_flags} %{link_flags} -DSHARED -fPIC -fvisibility=hidden -shared -o %t.shared_lib
// RUN: %{build} -fvisibility=hidden %t.shared_lib
// RUN: %{run}

#include <algorithm>
#include <cassert>
#include <type_traits>
#include <vector>

[[gnu::visibility("default")]] extern std::vector<void*> shared_lib_ptrs;
[[gnu::visibility("default")]] extern std::vector<const std::type_info*> shared_lib_type_infos;

inline std::vector<void*> get_ptrs() {
  return {
      // [meta.unary.cat]
      (void*)&std::is_void_v<int>,
      (void*)&std::is_null_pointer_v<int>,
      (void*)&std::is_integral_v<int>,
      (void*)&std::is_floating_point_v<int>,
      (void*)&std::is_array_v<int>,
      (void*)&std::is_pointer_v<int>,
      (void*)&std::is_lvalue_reference_v<int>,
      (void*)&std::is_rvalue_reference_v<int>,
      (void*)&std::is_member_object_pointer_v<int>,
      (void*)&std::is_member_function_pointer_v<int>,
      (void*)&std::is_enum_v<int>,
      (void*)&std::is_union_v<int>,
      (void*)&std::is_class_v<int>,
      (void*)&std::is_function_v<int>,

      // [meta.unary.comp]
      (void*)&std::is_reference_v<int>,
      (void*)&std::is_arithmetic_v<int>,
      (void*)&std::is_fundamental_v<int>,
      (void*)&std::is_object_v<int>,
      (void*)&std::is_scalar_v<int>,
      (void*)&std::is_compound_v<int>,
      (void*)&std::is_member_pointer_v<int>,

      // [meta.unary.prop]
      (void*)&std::is_const_v<int>,
      (void*)&std::is_volatile_v<int>,
      (void*)&std::is_trivial_v<int>,
      (void*)&std::is_trivially_copyable_v<int>,
      (void*)&std::is_standard_layout_v<int>,
      (void*)&std::is_empty_v<int>,
      (void*)&std::is_polymorphic_v<int>,
      (void*)&std::is_abstract_v<int>,
      (void*)&std::is_final_v<int>,
      (void*)&std::is_aggregate_v<int>,
      (void*)&std::is_signed_v<int>,
      (void*)&std::is_unsigned_v<int>,
      (void*)&std::is_bounded_array_v<int>,
      (void*)&std::is_unbounded_array_v<int>,
      (void*)&std::is_scoped_enum_v<int>,
      (void*)&std::is_constructible_v<int>,
      (void*)&std::is_default_constructible_v<int>,
      (void*)&std::is_copy_constructible_v<int>,
      (void*)&std::is_move_constructible_v<int>,
      (void*)&std::is_assignable_v<int, int>,
      (void*)&std::is_copy_assignable_v<int>,
      (void*)&std::is_move_assignable_v<int>,
      (void*)&std::is_swappable_with_v<int, int>,
      (void*)&std::is_swappable_v<int>,
      (void*)&std::is_destructible_v<int>,
      (void*)&std::is_trivially_constructible_v<int>,
      (void*)&std::is_trivially_default_constructible_v<int>,
      (void*)&std::is_trivially_copy_constructible_v<int>,
      (void*)&std::is_trivially_move_constructible_v<int>,
      (void*)&std::is_trivially_assignable_v<int, int>,
      (void*)&std::is_trivially_copy_assignable_v<int>,
      (void*)&std::is_trivially_move_assignable_v<int>,
      (void*)&std::is_trivially_destructible_v<int>,
      (void*)&std::is_nothrow_constructible_v<int, int>,
      (void*)&std::is_nothrow_default_constructible_v<int>,
      (void*)&std::is_nothrow_copy_constructible_v<int>,
      (void*)&std::is_nothrow_move_constructible_v<int>,
      (void*)&std::is_nothrow_assignable_v<int, int>,
      (void*)&std::is_nothrow_copy_assignable_v<int>,
      (void*)&std::is_nothrow_move_assignable_v<int>,
      (void*)&std::is_nothrow_swappable_with_v<int, int>,
      (void*)&std::is_nothrow_swappable_v<int>,
      (void*)&std::is_nothrow_destructible_v<int>,
#if 0
      (void*)&std::is_implicit_lifetime_v<int>,
#endif
      (void*)&std::has_virtual_destructor_v<int>,
      (void*)&std::has_unique_object_representations_v<int>,
#if 0
      (void*)&std::reference_constructs_from_temporary_v<int>,
      (void*)&std::reference_converts_from_temporary_v<int>,
#endif
  };
}

inline std::vector<const std::type_info*> get_type_infos() {
    return {
      // [meta.unary.cat]
      &typeid(std::is_void<int>),
      &typeid(std::is_null_pointer<int>),
      &typeid(std::is_integral<int>),
      &typeid(std::is_floating_point<int>),
      &typeid(std::is_array<int>),
      &typeid(std::is_pointer<int>),
      &typeid(std::is_lvalue_reference<int>),
      &typeid(std::is_rvalue_reference<int>),
      &typeid(std::is_member_object_pointer<int>),
      &typeid(std::is_member_function_pointer<int>),
      &typeid(std::is_enum<int>),
      &typeid(std::is_union<int>),
      &typeid(std::is_class<int>),
      &typeid(std::is_function<int>),

      // [meta.unary.comp]
      &typeid(std::is_reference_v<int>),
      &typeid(std::is_arithmetic_v<int>),
      &typeid(std::is_fundamental_v<int>),
      &typeid(std::is_object_v<int>),
      &typeid(std::is_scalar_v<int>),
      &typeid(std::is_compound_v<int>),
      &typeid(std::is_member_pointer_v<int>),

      // [meta.unary.prop]
      &typeid(std::is_const<int>),
      &typeid(std::is_volatile<int>),
      &typeid(std::is_trivial<int>),
      &typeid(std::is_trivially_copyable<int>),
      &typeid(std::is_standard_layout<int>),
      &typeid(std::is_empty<int>),
      &typeid(std::is_polymorphic<int>),
      &typeid(std::is_abstract<int>),
      &typeid(std::is_final<int>),
      &typeid(std::is_aggregate<int>),
      &typeid(std::is_signed<int>),
      &typeid(std::is_unsigned<int>),
      &typeid(std::is_bounded_array<int>),
      &typeid(std::is_unbounded_array<int>),
      &typeid(std::is_scoped_enum<int>),
      &typeid(std::is_constructible<int>),
      &typeid(std::is_default_constructible<int>),
      &typeid(std::is_copy_constructible<int>),
      &typeid(std::is_move_constructible<int>),
      &typeid(std::is_assignable<int, int>),
      &typeid(std::is_copy_assignable<int>),
      &typeid(std::is_move_assignable<int>),
      &typeid(std::is_swappable_with<int, int>),
      &typeid(std::is_swappable<int>),
      &typeid(std::is_destructible<int>),
      &typeid(std::is_trivially_constructible<int>),
      &typeid(std::is_trivially_default_constructible<int>),
      &typeid(std::is_trivially_copy_constructible<int>),
      &typeid(std::is_trivially_move_constructible<int>),
      &typeid(std::is_trivially_assignable<int, int>),
      &typeid(std::is_trivially_copy_assignable<int>),
      &typeid(std::is_trivially_move_assignable<int>),
      &typeid(std::is_trivially_destructible<int>),
      &typeid(std::is_nothrow_constructible<int, int>),
      &typeid(std::is_nothrow_default_constructible<int>),
      &typeid(std::is_nothrow_copy_constructible<int>),
      &typeid(std::is_nothrow_move_constructible<int>),
      &typeid(std::is_nothrow_assignable<int, int>),
      &typeid(std::is_nothrow_copy_assignable<int>),
      &typeid(std::is_nothrow_move_assignable<int>),
      &typeid(std::is_nothrow_swappable_with<int, int>),
      &typeid(std::is_nothrow_swappable<int>),
      &typeid(std::is_nothrow_destructible<int>),
#if 0
      &typeid(std::is_implicit_lifetime<int>),
#endif
      &typeid(std::has_virtual_destructor<int>),
      &typeid(std::has_unique_object_representations<int>),
#if 0
      &typeid(std::reference_constructs_from_temporary<int>),
      &typeid(std::reference_converts_from_temporary<int>),
#endif
  };
}

#ifdef SHARED
std::vector<void*> shared_lib_ptrs                       = get_ptrs();
std::vector<const std::type_info*> shared_lib_type_infos = get_type_infos();
#else
int main(int, char**) {
  assert(get_ptrs() == shared_lib_ptrs);
  auto deref = [](auto ptr) -> decltype(auto) { return *ptr; };
  assert(std::ranges::equal(get_type_infos(), shared_lib_type_infos, {}, deref, deref));

  return 0;
}
#endif
