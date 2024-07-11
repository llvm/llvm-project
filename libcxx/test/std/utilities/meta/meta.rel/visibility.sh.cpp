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
      (void*)&std::is_same_v<int, int>,
      (void*)&std::is_base_of_v<int, int>,
      (void*)&std::is_virtual_base_of_v<int, int>,
      (void*)&std::is_convertible_v<int, int>,
      (void*)&std::is_nothrow_convertible_v<int, int>,
#if 0
      (void*)&std::is_layout_compatible_v<int, int>,
      (void*)&std::is_pointer_interconvertible_base_of_v<int, int>,
#endif
      (void*)&std::is_invocable_v<int, int>,
      (void*)&std::is_invocable_r_v<int, int>,
      (void*)&std::is_nothrow_invocable_v<int, int>,
      (void*)&std::is_nothrow_invocable_r_v<int, int>,
  };
}

inline std::vector<const std::type_info*> get_type_infos() {
  return {
      &typeid(std::is_same<int, int>),
      &typeid(std::is_base_of<int, int>),
      &typeid(std::is_virtual_base_of<int, int>),
      &typeid(std::is_convertible<int, int>),
      &typeid(std::is_nothrow_convertible<int, int>),
#if 0
      &typeid(std::is_layout_compatible<int, int>),
      &typeid(std::is_pointer_interconvertible_base_of<int, int>),
#endif
      &typeid(std::is_invocable<int, int>),
      &typeid(std::is_invocable_r<int, int>),
      &typeid(std::is_nothrow_invocable<int, int>),
      &typeid(std::is_nothrow_invocable_r<int, int>),
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
