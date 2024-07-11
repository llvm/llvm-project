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

[[gnu::visibility("default")]] extern std::vector<const std::type_info*> shared_lib_type_infos;

struct [[gnu::visibility("default")]] S {
  static constexpr bool value = false;
};

inline std::vector<const std::type_info*> get_type_infos() {
  return {
      &typeid(std::remove_const<int>),
      &typeid(std::remove_volatile<int>),
      &typeid(std::remove_cv<int>),
      &typeid(std::add_const<int>),
      &typeid(std::add_volatile<int>),
      &typeid(std::add_cv<int>),
      &typeid(std::remove_reference<int>),
      &typeid(std::add_lvalue_reference<int>),
      &typeid(std::add_rvalue_reference<int>),
      &typeid(std::make_signed<int>),
      &typeid(std::make_unsigned<int>),
      &typeid(std::remove_extent<int>),
      &typeid(std::remove_all_extents<int>),
      &typeid(std::remove_pointer<int>),
      &typeid(std::add_pointer<int>),
      &typeid(std::type_identity<int>),
      &typeid(std::remove_cvref<int>),
      &typeid(std::decay<int>),
      &typeid(std::enable_if<true>),
      &typeid(std::conditional<true, int, int>),
      &typeid(std::common_type<int>),
      &typeid(std::common_reference<int>),
      &typeid(std::underlying_type<int>),
      &typeid(std::invoke_result<int>),
      &typeid(std::unwrap_reference<int>),
      &typeid(std::unwrap_ref_decay<int>),
      &typeid(std::conjunction<S>),
      &typeid(std::disjunction<S>),
      &typeid(std::negation<S>),
  };
}

#ifdef SHARED
std::vector<const std::type_info*> shared_lib_type_infos = get_type_infos();
#else
int main(int, char**) {
  auto deref = [](auto ptr) -> decltype(auto) { return *ptr; };
  assert(std::ranges::equal(get_type_infos(), shared_lib_type_infos, {}, deref, deref));

  return 0;
}
#endif
