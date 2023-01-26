//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12|13|14|15}}
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx{{11.0|12.0}}

// <vector>

// namespace std::pmr {
// template <class T>
// using vector =
//     ::std::vector<T, polymorphic_allocator<T>>
//
// } // namespace std::pmr

#include <vector>
#include <memory_resource>
#include <type_traits>
#include <cassert>

int main(int, char**) {
  using StdVector = std::vector<int, std::pmr::polymorphic_allocator<int>>;
  using PmrVector = std::pmr::vector<int>;
  static_assert(std::is_same<StdVector, PmrVector>::value, "");
  PmrVector d;
  assert(d.get_allocator().resource() == std::pmr::get_default_resource());

  return 0;
}
