//===-- alignment.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gwp_asan/tests/harness.h"
#include "gwp_asan/utilities.h"

#include <vector>

TEST(AlignmentTest, PowerOfTwo) {
  std::vector<std::pair<size_t, size_t>> AskedSizeToAlignedSize = {
      {1, 1},   {2, 2},   {3, 4},       {4, 4},       {5, 8},   {7, 8},
      {8, 8},   {9, 16},  {15, 16},     {16, 16},     {17, 32}, {31, 32},
      {32, 32}, {33, 48}, {4095, 4096}, {4096, 4096},
  };

  for (const auto &KV : AskedSizeToAlignedSize) {
    EXPECT_EQ(KV.second,
              gwp_asan::rightAlignedAllocationSize(
                  KV.first, gwp_asan::AlignmentStrategy::POWER_OF_TWO));
  }
}

TEST(AlignmentTest, AlignBionic) {
  std::vector<std::pair<size_t, size_t>> AskedSizeToAlignedSize = {
      {1, 8},   {2, 8},   {3, 8},       {4, 8},       {5, 8},   {7, 8},
      {8, 8},   {9, 16},  {15, 16},     {16, 16},     {17, 24}, {31, 32},
      {32, 32}, {33, 40}, {4095, 4096}, {4096, 4096},
  };

  for (const auto &KV : AskedSizeToAlignedSize) {
    EXPECT_EQ(KV.second, gwp_asan::rightAlignedAllocationSize(
                             KV.first, gwp_asan::AlignmentStrategy::BIONIC));
  }
}

TEST(AlignmentTest, PerfectAlignment) {
  for (size_t i = 1; i <= 4096; ++i) {
    EXPECT_EQ(i, gwp_asan::rightAlignedAllocationSize(
                     i, gwp_asan::AlignmentStrategy::PERFECT));
  }
}
