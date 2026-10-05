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

TEST(Handler, Prefetch) {
  mock::MockWrapper Mock;
  sycl::queue Q;

  int *Ptr = reinterpret_cast<int *>(1);
  constexpr std::size_t NumBytes = 1024;
  constexpr ol_mem_migration_flags_t ExpectedFlag =
      OL_MEM_MIGRATION_FLAG_HOST_TO_DEVICE;

  EXPECT_CALL(Mock.get(), olMemPrefetch(_, 1, _, _, ExpectedFlag))
      .Times(1)
      .WillRepeatedly([&](ol_queue_handle_t Queue, size_t Count,
                          const void **Mems, const size_t *Sizes,
                          ol_mem_migration_flags_t Flags) -> ol_result_t {
        EXPECT_NE(Queue, nullptr);
        EXPECT_EQ(Mems[0], Ptr);
        EXPECT_EQ(Sizes[0], NumBytes);
        return OL_SUCCESS;
      });
  auto E = Q.submit([&](sycl::handler &CGH) { CGH.prefetch(Ptr, NumBytes); });
}

TEST(Handler, DependsOnWithPrefetch) {
  mock::MockWrapper Mock;
  sycl::queue Q;

  int *Ptr = reinterpret_cast<int *>(1);
  constexpr std::size_t NumBytes = 1024;
  constexpr ol_mem_migration_flags_t ExpectedFlag =
      OL_MEM_MIGRATION_FLAG_HOST_TO_DEVICE;

  EXPECT_CALL(Mock.get(), olMemPrefetch(_, 1, _, _, ExpectedFlag)).Times(2);
  EXPECT_CALL(Mock.get(), olWaitEvents(_, _, 1)).Times(1);
  auto E = Q.submit([&](sycl::handler &CGH) { CGH.prefetch(Ptr, NumBytes); });
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(E);
    CGH.prefetch(Ptr, NumBytes);
  });
}
