//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <mock/helpers.hpp>

#include <sycl/__impl/queue.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace ::testing;

TEST(Handler, FIll) {
  mock::MockWrapper Mock;
  sycl::queue Q;

  int *Ptr = reinterpret_cast<int *>(1);
  int Pattern = 42;
  constexpr int FillCount = 32;
  constexpr std::size_t FillBytes = FillCount * sizeof(int);

  EXPECT_CALL(Mock.get(), olMemFill(_, Ptr, sizeof(int), _, FillBytes))
      .Times(1);
  auto E =
      Q.submit([&](sycl::handler &CGH) { CGH.fill(Ptr, Pattern, FillCount); });
}

TEST(Handler, DependsOnWithFill) {
  mock::MockWrapper Mock;
  sycl::queue Q;

  int *Ptr = reinterpret_cast<int *>(1);
  int Pattern = 42;
  constexpr int FillCount = 32;
  constexpr std::size_t FillBytes = FillCount * sizeof(int);

  EXPECT_CALL(Mock.get(), olMemFill(_, Ptr, sizeof(int), _, FillBytes))
      .Times(2);
  EXPECT_CALL(Mock.get(), olWaitEvents(_, _, 1)).Times(1);
  auto E =
      Q.submit([&](sycl::handler &CGH) { CGH.fill(Ptr, Pattern, FillCount); });
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(E);
    CGH.fill(Ptr, Pattern, FillCount);
  });
}
