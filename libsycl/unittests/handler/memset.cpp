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

TEST(Handler, Memset) {
  mock::MockWrapper Mock;
  sycl::queue Q;

  int *Ptr = reinterpret_cast<int *>(1);
  constexpr int NumBytes = 32;
  int Pattern = 42;

  EXPECT_CALL(Mock.get(), olMemFill(_, Ptr, sizeof(unsigned char), _, NumBytes))
      .Times(1);
  Q.submit([&](sycl::handler &CGH) { CGH.memset(Ptr, Pattern, NumBytes); });
}

TEST(Handler, DependsOnWithMemset) {
  mock::MockWrapper Mock;
  sycl::queue Q;

  int *Ptr = reinterpret_cast<int *>(1);
  constexpr int NumBytes = 32;
  int Pattern = 42;

  EXPECT_CALL(Mock.get(), olMemFill(_, Ptr, sizeof(unsigned char), _, NumBytes))
      .Times(2);
  EXPECT_CALL(Mock.get(), olWaitEvents(_, _, 1)).Times(1);
  auto E =
      Q.submit([&](sycl::handler &CGH) { CGH.memset(Ptr, Pattern, NumBytes); });
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(E);
    CGH.memset(Ptr, Pattern, NumBytes);
  });
}
