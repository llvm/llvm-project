//===-- flang/unittests/Runtime/Random.cpp ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime//random.h"
#include "gtest/gtest.h"
#include "flang/Runtime/descriptor.h"
#include "flang/Runtime/type-code.h"
#include <cmath>

using namespace Fortran::runtime;

TEST(RandomNumber, Real4) {
  StaticDescriptor<1> statDesc;
  Descriptor &harvest{statDesc.descriptor()};
  static constexpr int n{10000};
  float xs[n]{};
  SubscriptValue extent[1]{n};
  harvest.Establish(TypeCategory::Real, 4, xs, 1, extent);
  RTNAME(RandomNumber)(harvest, __FILE__, __LINE__);
  double sum{0};
  for (int j{0}; j < n; ++j) {
    sum += xs[j];
  }
  double mean{sum / n};
  std::fprintf(stderr, "mean of %d random numbers: %g\n", n, mean);
  EXPECT_GE(mean, 0.95 * 0.5); // mean of uniform dist [0..1] is of course 0.5
  EXPECT_LE(mean, 1.05 * 0.5);
  double sumsq{0};
  for (int j{0}; j < n; ++j) {
    double diff{xs[j] - mean};
    sumsq += diff * diff;
  }
  double sdev{std::sqrt(sumsq / n)};
  std::fprintf(stderr, "stddev of %d random numbers: %g\n", n, sdev);
  double expect{1.0 / std::sqrt(12.0)}; // stddev of uniform dist [0..1]
  EXPECT_GE(sdev, 0.95 * expect);
  EXPECT_LT(sdev, 1.05 * expect);
}

TEST(RandomNumber, RandomSeed) {
  StaticDescriptor<1> statDesc[2];
  Descriptor &desc{statDesc[0].descriptor()};
  std::int32_t n;
  desc.Establish(TypeCategory::Integer, 4, &n, 0, nullptr);
  RTNAME(RandomSeedSize)(&desc, __FILE__, __LINE__);
  EXPECT_EQ(n, 1);
  SubscriptValue extent[1]{1};
  desc.Establish(TypeCategory::Integer, 4, &n, 1, extent);
  RTNAME(RandomSeedGet)(&desc, __FILE__, __LINE__);
  Descriptor &harvest{statDesc[1].descriptor()};
  float x;
  harvest.Establish(TypeCategory::Real, 4, &x, 1, extent);
  RTNAME(RandomNumber)(harvest, __FILE__, __LINE__);
  float got{x};
  RTNAME(RandomSeedPut)(&desc, __FILE__, __LINE__); // n from RandomSeedGet()
  RTNAME(RandomNumber)(harvest, __FILE__, __LINE__);
  EXPECT_EQ(x, got);
}
