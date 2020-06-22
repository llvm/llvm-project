//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// template <class Alloc>
//   EXPLICIT tuple(allocator_arg_t, const Alloc& a, const Types&...);

// UNSUPPORTED: c++03

#include <tuple>
#include <memory>
#include <cassert>

struct ExplicitCopy {
  explicit ExplicitCopy(ExplicitCopy const&) {}
  explicit ExplicitCopy(int) {}
};

std::tuple<ExplicitCopy> const_explicit_copy_test() {
    const ExplicitCopy e(42);
    return {std::allocator_arg, std::allocator<int>{}, e};
    // expected-error@-1 {{chosen constructor is explicit in copy-initialization}}
}

std::tuple<ExplicitCopy> non_const_explicity_copy_test() {
    ExplicitCopy e(42);
    return {std::allocator_arg, std::allocator<int>{}, e};
    // expected-error@-1 {{chosen constructor is explicit in copy-initialization}}
}
int main(int, char**)
{
    const_explicit_copy_test();
    non_const_explicity_copy_test();

  return 0;
}
