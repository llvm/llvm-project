//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: c++11, c++14, c++17, c++20, c++23, c++26, c++2b, c++2c
//
// This test verifies that the std::random_shuffle overload taking a
// random-number generator function (legacy) is available and callable
// in dialects where random_shuffle is still provided ( == C++03).
//
// Compile-only, so it can run under c++03 as well.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <vector>
#include <iterator>

typedef std::vector<int>::iterator Iter;
typedef std::iterator_traits<Iter>::difference_type Diff;
typedef Diff (*RandomNumberGenerator)(Diff);

// Wrapper that calls the legacy random_shuffle overload with RNG by reference.
void f(Iter a1, Iter a2, RandomNumberGenerator& a3) { std::random_shuffle<Iter, RandomNumberGenerator>(a1, a2, a3); }

// A minimal RNG with the required signature. Behavior doesn't matter for compile test.
static Diff rng(Diff n) {
  return (n <= 1) ? 0 : (n - 1); // returns in [0, n)
}

int main() {
  // Use a C array to avoid C++11 initializer_list; this is valid in C++03.
  int data[] = {0, 1, 2, 3, 4};
  std::vector<int> v(data, data + sizeof(data) / sizeof(data[0]));

  RandomNumberGenerator g = &rng;
  f(v.begin(), v.end(), g);
  return 0;
}
