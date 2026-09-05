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

TEST(Queue, Memset) {
  mock::MockWrapper Mock;
  sycl::queue Q;

  int a;
  int *Ptr = &a;
  constexpr int FillCount = 32;
  int Pattern = 42;

  EXPECT_CALL(Mock.get(),
              olMemFill(_, Ptr, sizeof(unsigned char), _, FillCount))
      .Times(3);
  EXPECT_CALL(Mock.get(), olWaitEvents(_, _, 1)).Times(2);
  sycl::event E = Q.memset(Ptr, Pattern, FillCount);
  Q.memset(Ptr, Pattern, FillCount, E);
  Q.memset(Ptr, Pattern, FillCount, std::vector<sycl::event>{E});
}

TEST(Queue, MemsetZeroBytes) {
  mock::MockWrapper Mock;
  sycl::queue Q;
  EXPECT_CALL(Mock.get(), olWaitEvents(_, _, 1)).Times(2);
  EXPECT_CALL(Mock.get(), olMemFill(_, _, _, _, _)).Times(0);
  sycl::event E = Q.memset(nullptr, 1, 0);
  Q.memset(nullptr, 1, 0, E);
  Q.memset(nullptr, 1, 0, std::vector<sycl::event>{E});
}
