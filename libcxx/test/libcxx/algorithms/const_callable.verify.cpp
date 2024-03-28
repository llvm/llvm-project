//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <algorithm>

#include <algorithm>

struct ConstUncallable {
  template <class T>
  bool operator()(const T& x, const T& y) & {
    return x < y;
  }
  template <class T>
  bool operator()(const T& x, const T& y) const& = delete;
};

struct NonConstUncallable {
  template <class T>
  bool operator()(const T& x, const T& y) const& {
    return x < y;
  }
  template <class T>
  bool operator()(const T& x, const T& y) & = delete;
};

void test() {
  {
    auto pair =
        std::minmax({0, 1}, ConstUncallable{}); // expected-error@*:* {{The comparator has to be const-callable}}
    (void)pair;
  }
}
