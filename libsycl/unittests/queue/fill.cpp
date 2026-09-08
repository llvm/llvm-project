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

#include <cstdint>

using namespace ::testing;

TEST(Queue, Fill) {
  mock::MockWrapper Mock;
  sycl::queue Q;

  int *Ptr = reinterpret_cast<int *>(1);
  int Pattern = 42;
  int *PatternPtr = &Pattern;
  constexpr int FillCount = 32;
  constexpr std::size_t FillBytes = FillCount * sizeof(int);

  EXPECT_CALL(Mock.get(), olMemFill(_, Ptr, sizeof(int), PatternPtr, FillBytes))
      .Times(3);
  EXPECT_CALL(Mock.get(), olWaitEvents(_, _, 1)).Times(2);
  sycl::event E = Q.fill(Ptr, Pattern, FillCount);
  Q.fill(Ptr, Pattern, FillCount, E);
  Q.fill(Ptr, Pattern, FillCount, std::vector<sycl::event>{E});
}

TEST(Queue, FillZeroCount) {
  mock::MockWrapper Mock;
  sycl::queue Q;
  EXPECT_CALL(Mock.get(), olWaitEvents(_, _, 1)).Times(2);
  EXPECT_CALL(Mock.get(), olMemFill(_, _, _, _, _)).Times(0);
  sycl::event E = Q.fill(nullptr, 1, 0);
  Q.fill(nullptr, 1, 0, E);
  Q.fill(nullptr, 1, 0, std::vector<sycl::event>{E});
}

TEST(Queue, FillNullptr) {
  mock::MockWrapper Mock;
  sycl::queue Q;
  sycl::event Dep = Q.fill(nullptr, 1, 0);
  EXPECT_CALL(Mock.get(), olWaitEvents(_, _, _)).Times(0);
  EXPECT_CALL(Mock.get(), olMemFill(_, _, _, _, _)).Times(0);
  try {
    Q.fill(nullptr, 1, 1);
    FAIL() << "Expected thrown exception";
  } catch (sycl::exception &E) {
    EXPECT_EQ(E.code(), make_error_code(sycl::errc::invalid));
    EXPECT_NE(std::string(E.what()).find("Nullptr argument"),
              std::string::npos);
  }
}

TEST(Queue, FillBytesGTSizeMax) {
  mock::MockWrapper Mock;
  sycl::queue Q;
  int *Ptr = reinterpret_cast<int *>(1);
  sycl::event Dep = Q.fill(Ptr, 1, 0);
  EXPECT_CALL(Mock.get(), olWaitEvents(_, _, _)).Times(0);
  EXPECT_CALL(Mock.get(), olMemFill(_, _, _, _, _)).Times(0);
  try {
    Q.fill(Ptr, 1, SIZE_MAX / sizeof(int) + 1);
    FAIL() << "Expected thrown exception";
  } catch (sycl::exception &E) {
    EXPECT_EQ(E.code(), make_error_code(sycl::errc::invalid));
    EXPECT_NE(std::string(E.what()).find(
                  "Total number of bytes to be filled exceeds SIZE_MAX"),
              std::string::npos);
  }
}