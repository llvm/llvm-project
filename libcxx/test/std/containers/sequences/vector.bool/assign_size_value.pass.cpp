//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// void assign(size_type n, const value_type& x);

#include <vector>
#include <cassert>
#include "test_macros.h"
#include "test_iterators.h"

TEST_CONSTEXPR_CXX20 bool tests() {
  {   // Test with various cases where assign may or may not trigger reallocations
    { // Reallocation happens
      std::size_t N = 128;
      std::vector<bool> v(5, false);
      assert(v.capacity() < N);
      v.assign(N, true);
      assert(v.size() == N);
      for (std::size_t i = 0; i < N; ++i)
        assert(v[i] == true);
    }
    { // No reallocation: fit within current size
      std::size_t N = 5;
      std::vector<bool> v(2 * N, false);
      v.assign(N, true);
      assert(v.size() == N);
      for (std::size_t i = 0; i < N; ++i)
        assert(v[i] == true);
    }
    { // No reallocation: fit within spare space
      std::size_t N = 5;
      std::vector<bool> v(N / 2, false);
      v.reserve(N * 2);
      v.assign(N, true);
      assert(v.size() == N);
      for (std::size_t i = 0; i < N; ++i)
        assert(v[i] == true);
    }
  }

  return true;
}

int main(int, char**) {
  tests();
#if TEST_STD_VER > 17
  static_assert(tests());
#endif
  return 0;
}
