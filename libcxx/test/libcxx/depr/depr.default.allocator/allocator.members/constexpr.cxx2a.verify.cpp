//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

//  In C++20, parts of std::allocator<T> have been removed.
//  However, for backwards compatibility, if _LIBCPP_ENABLE_CXX20_REMOVED_ALLOCATOR_MEMBERS
//  is defined before including <memory>, then removed members will be restored.

//  This leads to `vector` and `string` using those members instead of `allocator_traits`.
//  So the restored members must also be made `constexpr` to make sure instantiated members
//  of those types stay `constexpr`.

//  Check that the necessary members, vector and string can be used as constants in this mode.

// UNSUPPORTED: c++03, c++11, c++14, c++17
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_CXX20_REMOVED_ALLOCATOR_MEMBERS
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS
// expected-no-diagnostics

#include <memory>
#include <vector>
#include <string>

static_assert(std::allocator<int>().max_size() != 0);

constexpr int check_lifetime_return_123() {
  std::allocator<int> a;
  int* p = a.allocate(10);
  a.construct(p, 1);
  a.destroy(p);
  a.deallocate(p, 10);
  return 123;
}
static_assert(check_lifetime_return_123() == 123);

static_assert(std::vector<int>({1, 2, 3}).size() == 3);
static_assert(std::string("abc").size() == 3);
