#include <mock/helpers.hpp>

#include <sycl/__impl/queue.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace sycl;
using namespace ::testing;

TEST(Queue, Fill) {
  mock::MockWrapper Mock;
  queue Q;

  int a;
  int *Ptr = &a;
  int Pattern;
  int *PatternPtr = &Pattern;
  constexpr int FillCount = 32;
  constexpr int FillBytes = FillCount * sizeof(int);

  EXPECT_CALL(Mock.get(), olMemFill(_, Ptr, sizeof(int), PatternPtr, FillBytes))
      .Times(3);
  EXPECT_CALL(Mock.get(), olWaitEvents(_, _, 1)).Times(2);
  event E = Q.fill(Ptr, Pattern, FillCount);
  Q.fill(Ptr, Pattern, FillCount, E);
  Q.fill(Ptr, Pattern, FillCount, std::vector<event>{E});
}

TEST(Queue, FillZeroBytes) {
  mock::MockWrapper Mock;
  queue Q;
  EXPECT_CALL(Mock.get(), olWaitEvents(_, _, 1)).Times(2);
  EXPECT_CALL(Mock.get(), olMemFill(_, _, _, _, _)).Times(0);
  event E = Q.fill(nullptr, 1, 0);
  Q.fill(nullptr, 1, 0, E);
  Q.fill(nullptr, 1, 0, std::vector<event>{E});
}
