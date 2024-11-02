//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <random>

#include <random>

template<class Int>
struct G {
  using result_type = Int;
  result_type operator()();
  static constexpr result_type min() { return 0; }
  static constexpr result_type max() { return 255; }
};

void test(std::normal_distribution<double> dist)
{
  G<int> badg;
  G<unsigned> okg;

  dist(badg); //expected-error-re@*:* 2 {{{{(static_assert|static assertion)}} failed}} //expected-note {{in instantiation}}
  dist(okg);
}
