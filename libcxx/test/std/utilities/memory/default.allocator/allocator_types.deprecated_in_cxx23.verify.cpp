//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: c++23

// <memory>

// template <class T>
// class allocator
// {
// ...
//     typedef true_type is_always_equal; // Deprecated in C++23, removed in C++26
// ...
// };

#include <memory>

void test() {
  {
    typedef std::allocator<char>::is_always_equal IAE; // expected-warning {{'is_always_equal' is deprecated}}
  }
  {
    typedef std::allocator<int>::is_always_equal IAE; // expected-warning {{'is_always_equal' is deprecated}}
  }
  {
    typedef std::allocator<void>::is_always_equal IAE; // expected-warning {{'is_always_equal' is deprecated}}
  }
}
