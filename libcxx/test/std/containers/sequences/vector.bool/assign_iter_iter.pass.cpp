//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// void assign(_InputIterator __first, _InputIterator __last);
// void assign(_ForwardIterator __first, _ForwardIterator __last);

#include <vector>
#include <cassert>
#include "test_macros.h"
#include "test_iterators.h"

TEST_CONSTEXPR_CXX20 bool tests() {
  {   // Test with various cases where assign may or may not trigger reallocations for forward_iterator
    { // Reallocation happens
      std::vector<bool> in(128, true);
      std::vector<bool> v(5, false);
      assert(v.capacity() < in.size());
      using It = forward_iterator<std::vector<bool>::iterator>;
      v.assign(It(in.begin()), It(in.end()));
      assert(v == in);
    }
    { // No reallocation: fit wintin current size
      bool in[]                    = {false, true, false, true, true};
      TEST_CONSTEXPR std::size_t N = sizeof(in) / sizeof(in[0]);
      std::vector<bool> v(2 * N, false);
      using It = forward_iterator<bool*>;
      v.assign(It(in), It(in + N));
      assert(v.size() == N);
      for (std::size_t i = 0; i < N; ++i)
        assert(v[i] == in[i]);
    }
    { // No reallocation: fit wintin spare space
      bool in[]                    = {false, true, false, true, true};
      TEST_CONSTEXPR std::size_t N = sizeof(in) / sizeof(in[0]);
      std::vector<bool> v(N / 2, false);
      v.reserve(N * 2);
      using It = forward_iterator<bool*>;
      v.assign(It(in), It(in + N));
      assert(v.size() == N);
      for (std::size_t i = 0; i < N; ++i)
        assert(v[i] == in[i]);
    }
  }

  {   // Test with various cases where assign may or may not trigger reallocations for input_iterator
    { // Reallocation happens
      std::vector<bool> in(128, true);
      std::vector<bool> v(5, false);
      assert(v.capacity() < in.size());
      using It = cpp17_input_iterator<std::vector<bool>::iterator>;
      v.assign(It(in.begin()), It(in.end()));
      assert(v == in);
    }
    { // No reallocation: fit wintin current size
      bool in[]                    = {false, true, false, true, true};
      TEST_CONSTEXPR std::size_t N = sizeof(in) / sizeof(in[0]);
      std::vector<bool> v(2 * N, false);
      using It = cpp17_input_iterator<bool*>;
      v.assign(It(in), It(in + N));
      assert(v.size() == N);
      for (std::size_t i = 0; i < N; ++i)
        assert(v[i] == in[i]);
    }
    { // No reallocation: fit wintin spare space
      bool in[]                    = {false, true, false, true, true};
      TEST_CONSTEXPR std::size_t N = sizeof(in) / sizeof(in[0]);
      std::vector<bool> v(N / 2, false);
      v.reserve(N * 2);
      using It = cpp17_input_iterator<bool*>;
      v.assign(It(in), It(in + N));
      assert(v.size() == N);
      for (std::size_t i = 0; i < N; ++i)
        assert(v[i] == in[i]);
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
