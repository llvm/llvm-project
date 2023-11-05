//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test verifies that you can make string reductions with parallel
// algorithms

#include <algorithm>
#include <cassert>
#include <execution>
#include <vector>

int main(int, char**) {
  int length = 10000;
  std::vector<std::string> vec(length);
  for (int i = 0; i < length; i++) {
    std::string str = "";
    char a;
    for (int j = 0; j < 5; j++) {
      a = 56 + (((int)i * 5 + 3 * j) % 200);
      str += a;
    }
    vec[i] = str;
  }

  std::string result = std::transform_reduce(
      std::execution::par_unseq, vec.begin(), vec.end(), (std::string) "", std::plus{}, [](std::string& str) {
        return str;
      });
  for (auto v : vec) {
    assert(result.find(v) != std::string::npos && "Substring not found in deruced string.");
  }
  return 0;
}