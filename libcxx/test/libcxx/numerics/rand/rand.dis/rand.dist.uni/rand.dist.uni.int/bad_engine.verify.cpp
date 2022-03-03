//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03
// ADDITIONAL_COMPILE_FLAGS: -Wno-sign-compare

// <random>

#include <random>

template<class Int>
struct G {
  using result_type = Int;
  result_type operator()();
  static constexpr result_type min() { return 0; }
  static constexpr result_type max() { return 255; }
};

void test(std::uniform_int_distribution<int> dist)
{
  G<int> badg;
  G<unsigned> okg;

  dist(badg); //expected-error@*:* {{static_assert failed}} //expected-note {{in instantiation}}
  dist(okg);
}
