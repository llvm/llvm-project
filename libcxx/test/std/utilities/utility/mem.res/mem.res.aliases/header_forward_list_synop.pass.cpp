//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// TODO: Change to XFAIL once https://github.com/llvm/llvm-project/issues/40340 is fixed
// UNSUPPORTED: availability-pmr-missing

// <forward_list>

// namespace std::pmr {
// template <class T>
// using forward_list =
//     ::std::forward_list<T, polymorphic_allocator<T>>
//
// } // namespace std::pmr

#include <forward_list>
#include <memory_resource>
#include <type_traits>
#include <cassert>

int main(int, char**) {
  using StdForwardList = std::forward_list<int, std::pmr::polymorphic_allocator<int>>;
  using PmrForwardList = std::pmr::forward_list<int>;
  static_assert(std::is_same<StdForwardList, PmrForwardList>::value, "");
  PmrForwardList d;
  assert(d.get_allocator().resource() == std::pmr::get_default_resource());

  return 0;
}
