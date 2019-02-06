//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// template <class Alloc, class ...UTypes>
//   tuple(allocator_arg_t, const Alloc& a, tuple<UTypes...>&&);

// UNSUPPORTED: c++98, c++03

#include <tuple>
#include <memory>

struct ExplicitCopy {
  explicit ExplicitCopy(int) {}
  explicit ExplicitCopy(ExplicitCopy const&) {}
};

std::tuple<ExplicitCopy> explicit_move_test() {
    std::tuple<int> t1(42);
    return {std::allocator_arg, std::allocator<void>{}, std::move(t1)};
    // expected-error@-1 {{chosen constructor is explicit in copy-initialization}}
}

int main()
{

}
