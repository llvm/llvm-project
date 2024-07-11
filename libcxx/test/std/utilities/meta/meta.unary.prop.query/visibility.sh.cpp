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
      (void*)&std::alignment_of_v<int>,
      (void*)&std::extent_v<int>,
      (void*)&std::rank_v<int>,
  };
}

inline std::vector<const std::type_info*> get_type_infos() {
  return {
      &typeid(std::alignment_of<int>),
      &typeid(std::extent<int>),
      &typeid(std::rank<int>),
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
