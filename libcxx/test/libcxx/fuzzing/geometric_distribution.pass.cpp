// -*- C++ -*-
//===------------------------ unique_copy.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

#include <random>
#include <cstdint>

#include "fuzzer_test.h"

template <class Distribution>
int random_distribution_helper(const uint8_t *data, size_t size) {
  std::mt19937 engine;
  using ParamT = typename Distribution::param_type;
  if (size < sizeof(double))
    return 0;
  double Arg;
  memcpy(&Arg, data, sizeof(double));
  ParamT p(Arg);
  Distribution d(p);
  for (int I=0; I < 1000; ++I) {
    volatile auto res = d(engine);
    ((void)res);
  }
  return 0;
}

int FuzzRandom(const uint8_t *Data, size_t Size) {
  return random_distribution_helper<std::geometric_distribution<std::int16_t>>(Data, Size);
}
FUZZER_TEST(FuzzRandom);


