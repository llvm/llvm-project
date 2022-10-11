//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <memory_resource>

// monotonic_buffer_resource(monotonic_buffer_resource const&) = delete;
// monotonic_buffer_resource& operator=(monotonic_buffer_resource const&) = delete;

#include <memory_resource>
#include <type_traits>

int main(int, char**) {
  using MBR = std::pmr::monotonic_buffer_resource;
  static_assert(!std::is_copy_constructible_v<MBR>);
  static_assert(!std::is_move_constructible_v<MBR>);
  static_assert(!std::is_copy_assignable_v<MBR>);
  static_assert(!std::is_move_assignable_v<MBR>);

  return 0;
}
