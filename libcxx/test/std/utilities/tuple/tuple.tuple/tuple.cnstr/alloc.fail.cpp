//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <tuple>

// template <class... Types> class tuple;

// template <class Alloc>
//   explicit(see-below) tuple(allocator_arg_t, const Alloc& a);

// Make sure we get the explicit-ness of the constructor right.
// This is LWG 3158.

#include <tuple>
#include <memory>


struct ExplicitDefault { explicit ExplicitDefault() { } };

std::tuple<ExplicitDefault> explicit_default_test() {
    return {std::allocator_arg, std::allocator<int>()}; // expected-error {{chosen constructor is explicit in copy-initialization}}
}

int main(int, char**) {
    return 0;
}
