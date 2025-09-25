//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// TODO: Change to XFAIL once https://llvm.org/PR40995 is fixed
// UNSUPPORTED: availability-pmr-missing

// <memory_resource>

// template <class T> class polymorphic_allocator

// polymorphic_allocator operator=(polymorphic_allocator const &) = delete

#include <memory_resource>
#include <type_traits>
#include <cassert>

int main(int, char**) {
  typedef std::pmr::polymorphic_allocator<void> T;
  static_assert(!std::is_copy_assignable<T>::value, "");
  static_assert(!std::is_move_assignable<T>::value, "");

  return 0;
}
