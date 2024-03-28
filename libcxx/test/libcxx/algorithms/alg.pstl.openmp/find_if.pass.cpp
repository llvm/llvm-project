//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test verifies that std::find_if(std::execution::par_unseq,...) always
// finds the first entry in a vector matching the condition. If it was confused
// with std::any_of, it could return the indexes in a non-increasing order.

// UNSUPPORTED: c++03, c++11, c++14, gcc

// ADDITIONAL_COMPILE_FLAGS: --offload-arch=native

// REQUIRES: libcpp-pstl-backend-openmp

#include <algorithm>
#include <array>
#include <cassert>
#include <execution>
#include <vector>

template <class _Tp>
void check_find_if(_Tp& data) {
  const int len = data.end() - data.begin();
  // Decrementing the values in the test indices
  int idx[11] = {0, len / 10, len / 9, len / 8, len / 7, len / 6, len / 5, len / 4, len / 3, len / 2, len - 1};
  for (auto i : idx) {
    data[i] -= 1;
  };

  // Asserting that the minimas are found in the correct order
  for (auto i : idx) {
    auto found_min = std::find_if(
        std::execution::par_unseq, data.begin(), data.end(), [&](decltype(data[0])& n) -> bool { return n < 2; });
    assert(found_min == (data.begin() + i));
    // Incrementing the minimum, so the next one can be found
    (*found_min) += 1;
  }
}

int main(int, char**) {
  const int test_size = 10000;
  // Testing with vector of doubles
  {
    std::vector<double> v(test_size, 2.0);
    check_find_if(v);
  }
  // Testing with vector of integers
  {
    std::vector<int> v(test_size, 2);
    check_find_if(v);
  }
  // Testing with array of doubles
  {
    std::array<double, test_size> a;
    a.fill(2.0);
    check_find_if(a);
  }
  // Testing with array of integers
  {
    std::array<int, test_size> a;
    a.fill(2);
    check_find_if(a);
  }
  return 0;
}
