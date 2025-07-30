//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS(clang): -Wprivate-header

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

#include <__iterator/aliasing_iterator.h>
#include <cassert>

struct NonTrivial {
  int i_;

  NonTrivial(int i) : i_(i) {}
  NonTrivial(const NonTrivial& other) : i_(other.i_) {}

  NonTrivial& operator=(const NonTrivial& other) {
    i_ = other.i_;
    return *this;
  }

  ~NonTrivial() {}
};

int main(int, char**) {
  {
    NonTrivial arr[] = {1, 2, 3, 4};
    std::__aliasing_iterator<NonTrivial*, int> iter(arr);

    assert(*iter == 1);
    assert(iter[0] == 1);
    assert(iter[1] == 2);
    ++iter;
    assert(*iter == 2);
    assert(iter[-1] == 1);
    assert(iter.__base() == arr + 1);
    assert(iter == iter);
    assert(iter != (iter + 1));
  }

  return 0;
}
