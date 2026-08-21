#include <mock/helpers.hpp>

#include <sycl/__impl/queue.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace sycl;
using namespace ::testing;

TEST(Queue, Memset) {
  mock::MockWrapper Mock;
  queue Q;

  int a;
  int *Ptr = &a;
  constexpr int FillCount = 32;
  int Pattern = 42;

  EXPECT_CALL(Mock.get(),
              olMemFill(_, Ptr, sizeof(unsigned char), _, FillCount))
      .Times(3);
  EXPECT_CALL(Mock.get(), olWaitEvents(_, _, 1)).Times(2);
  event E = Q.memset(Ptr, Pattern, FillCount);
  Q.memset(Ptr, Pattern, FillCount, E);
  Q.memset(Ptr, Pattern, FillCount, std::vector<event>{E});
}

TEST(Queue, MemsetZeroBytes) {
  mock::MockWrapper Mock;
  queue Q;
  EXPECT_CALL(Mock.get(), olWaitEvents(_, _, 1)).Times(2);
  EXPECT_CALL(Mock.get(), olMemFill(_, _, _, _, _)).Times(0);
  event E = Q.memset(nullptr, 1, 0);
  Q.memset(nullptr, 1, 0, E);
  Q.memset(nullptr, 1, 0, std::vector<event>{E});
}
