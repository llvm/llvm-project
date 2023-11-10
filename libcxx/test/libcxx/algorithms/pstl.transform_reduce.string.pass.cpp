//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test verifies that you can make string reductions with parallel
// algorithms

// UNSUPPORTED: c++03, c++11, c++14, c++17, libcpp-has-no-incomplete-pstl

#include <algorithm>
#include <cassert>
#include <execution>
#include <numeric>
#include <string>
#include <vector>

template <class Execution, class Input>
void test_execution_policy(Execution& execution, Input& input) {
  std::string result = std::transform_reduce(
      execution, input.begin(), input.end(), (std::string) "", std::plus{}, [](std::string& str) { return str; });
  for (auto v : input) {
    assert(result.find(v) != std::string::npos && "Substring not found in reduuced string.");
  }
}

int main(int, char**) {
  int length = 10000;
  std::vector<std::string> vec(length);

  // Generating semi-random strings of length 5
  unsigned int seed = 17;
  for (int i = 0; i < length; i++) {
    std::string str = "";
    char a;
    for (int j = 0; j < 5; j++) {
      // Generating random unsigned int with a linear congruential generator
      seed = (214013 * seed + 11) % 4294967295;

      // Generating ASCII character from 33 (!) to 126 (~)
      a = ((unsigned int)33 + (seed % 93));
      str += a;
    }
    vec[i] = str;
  }

  test_execution_policy(std::execution::seq, vec);
  test_execution_policy(std::execution::unseq, vec);
  test_execution_policy(std::execution::par, vec);
  test_execution_policy(std::execution::par_unseq, vec);
  return 0;
}