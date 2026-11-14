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

#include <cstddef>

using namespace sycl;
using namespace ::testing;

TEST(Queue, TwoPrefetches) {
  constexpr std::size_t NumBytes = 1024;
  constexpr int NPrefetches = 2;

  mock::MockWrapper Mock;
  queue Q;

  const void *Ptr = reinterpret_cast<void *>(1);

  constexpr ol_mem_migration_flags_t ExpectedFlag =
      OL_MEM_MIGRATION_FLAG_HOST_TO_DEVICE;

  EXPECT_CALL(Mock.get(), olMemPrefetch(_, 1, _, _, ExpectedFlag))
      .Times(NPrefetches)
      .WillRepeatedly([&](ol_queue_handle_t Queue, size_t Count,
                          const void **Mems, const size_t *Sizes,
                          ol_mem_migration_flags_t Flags) -> ol_result_t {
        EXPECT_NE(Queue, nullptr);
        EXPECT_EQ(Mems[0], Ptr);
        EXPECT_EQ(Sizes[0], NumBytes);
        return OL_SUCCESS;
      });

  EXPECT_CALL(Mock.get(), olCreateEvent(_, _, _)).Times(NPrefetches);

  event Event = Q.prefetch(Ptr, NumBytes);

  // second prefetch depends on the first one
  EXPECT_CALL(Mock.get(), olWaitEvents(_, _, 1));
  Q.prefetch(Ptr, NumBytes, Event);
}

TEST(Queue, PrefetchZeroBytes) {
  mock::MockWrapper Mock;
  queue Q;

  EXPECT_CALL(Mock.get(), olMemPrefetch(_, _, _, _, _)).Times(0);
  EXPECT_CALL(Mock.get(), olWaitEvents(_, _, 1)).Times(1);

  event Event = Q.prefetch(nullptr, 0);
  Q.prefetch(nullptr, 0, Event);
}

TEST(Queue, PrefetchNullptrThrows) {
  constexpr std::size_t NumBytes = 1024;

  mock::MockWrapper Mock;
  queue Q;

  EXPECT_CALL(Mock.get(), olMemPrefetch(_, _, _, _, _)).Times(0);

  try {
    Q.prefetch(nullptr, NumBytes);
    FAIL() << "Expected sycl::exception";
  } catch (const sycl::exception &E) {
    EXPECT_EQ(E.code(), make_error_code(errc::invalid));
    EXPECT_TRUE(E.has_context());
    EXPECT_EQ(E.get_context(), Q.get_context());
  }
}
