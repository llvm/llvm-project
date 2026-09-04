#include "test_helpers.hpp"
#include <mock/helpers.hpp>

#include <detail/device_impl.hpp>

#include <sycl/__impl/device.hpp>
#include <sycl/__impl/queue.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <string>

using namespace sycl;
using namespace ::testing;

TEST(Handler, MultipleActionsRejected) {
  mock::MockWrapper Mock;
  queue Q;
  int Src = 1;
  int Dst = 0;

  bool Thrown = false;
  try {
    Q.submit([&](handler &CGH) {
      CGH.memcpy(&Dst, &Src, sizeof(int));
      CGH.memcpy(&Dst, &Src, sizeof(int));
    });
  } catch (const sycl::exception &E) {
    Thrown = true;
    EXPECT_NE(std::string(E.what()).find("multiple actions"),
              std::string::npos);
  }

  EXPECT_TRUE(Thrown);
}

TEST(Handler, DependsOnOnlyCommandGroup) {
  mock::MockWrapper Mock;
  queue Q;

  constexpr int NumBytes = sizeof(int);
  int Src = 1;
  int Dst = 0;
  ol_device_handle_t OLDev =
      detail::getSyclObjImpl(Q.get_device())->getOLHandle();

  sycl::unittests::expectDeviceMemoryInfo(Mock, {&Src, &Dst}, OLDev, 2);
  EXPECT_CALL(Mock.get(), olMemcpy(_, &Dst, OLDev, &Src, OLDev, NumBytes))
      .Times(1);
  EXPECT_CALL(Mock.get(), olCreateEvent(_, _, _)).Times(2);

  auto DepEvent = Q.memcpy(&Dst, &Src, NumBytes);

  EXPECT_CALL(Mock.get(), olWaitEvents(_, _, 1)).Times(1);
  auto E = Q.submit([&](handler &CGH) { CGH.depends_on(DepEvent); });

  EXPECT_CALL(Mock.get(), olSyncEvent(_)).Times(1);
  E.wait();
}

TEST(Handler, DependsOnVectorOverload) {
  constexpr int NumBytes = 16;

  mock::MockWrapper Mock;
  queue Q;

  int SrcA[NumBytes / sizeof(int)] = {};
  int Mid[NumBytes / sizeof(int)] = {};
  int Dst[NumBytes / sizeof(int)] = {};

  ol_device_handle_t OLDev =
      detail::getSyclObjImpl(Q.get_device())->getOLHandle();

  sycl::unittests::expectDeviceMemoryInfo(Mock, {SrcA, Mid, Dst}, OLDev, 4);

  EXPECT_CALL(Mock.get(), olMemcpy(_, Mid, OLDev, SrcA, OLDev, NumBytes))
      .Times(1);
  EXPECT_CALL(Mock.get(), olMemcpy(_, Dst, OLDev, Mid, OLDev, NumBytes))
      .Times(1);

  // Two events for the two memcpys, one for the dependency-only command group.
  EXPECT_CALL(Mock.get(), olCreateEvent(_, _, _)).Times(3);

  auto E1 = Q.memcpy(Mid, SrcA, NumBytes);
  auto E2 = Q.memcpy(Dst, Mid, NumBytes);

  EXPECT_CALL(Mock.get(), olWaitEvents(_, _, 2)).Times(1);
  auto E = Q.submit(
      [&](handler &CGH) { CGH.depends_on(std::vector<event>{E1, E2}); });

  EXPECT_CALL(Mock.get(), olSyncEvent(_)).Times(1);
  E.wait();
}

TEST(Handler, EmptyCommandGroupNoDependencies) {
  mock::MockWrapper Mock;
  queue Q;

  EXPECT_CALL(Mock.get(), olWaitEvents(_, _, _)).Times(0);
  EXPECT_CALL(Mock.get(), olCreateEvent(_, _, _)).Times(1);

  auto E = Q.submit([&](handler &CGH) { (void)CGH; });

  EXPECT_CALL(Mock.get(), olSyncEvent(_)).Times(1);
  E.wait();
}

TEST(Queue, SubmitCannotBeNested) {
  mock::MockWrapper Mock;
  queue Q;

  bool Thrown = false;
  try {
    Q.submit([&](handler &CGH) {
      (void)CGH;
      Q.submit([&](handler &Nested) {
        Nested.single_task<class UTNestedSubmitKernel>([]() {});
      });
    });
  } catch (const sycl::exception &E) {
    Thrown = true;
    EXPECT_NE(std::string(E.what()).find("cannot be nested"),
              std::string::npos);
  }

  EXPECT_TRUE(Thrown);
}

TEST(Handler, ParallelForNDRangeRejectsNonDivisibleRange) {
  mock::MockWrapper Mock;
  queue Q;

  bool Thrown = false;
  try {
    Q.submit([&](handler &CGH) {
      CGH.parallel_for<class UTNDRangeInvalidNonDivisible>(
          nd_range<1>{range<1>{10}, range<1>{3}}, [=](nd_item<1>) {});
    });
  } catch (const sycl::exception &E) {
    Thrown = true;
    EXPECT_EQ(E.code(), sycl::errc::nd_range);
  }

  EXPECT_TRUE(Thrown);
}

TEST(Handler, ParallelForNDRangeRejectsZeroLocalRange) {
  mock::MockWrapper Mock;
  queue Q;

  bool Thrown = false;
  try {
    Q.submit([&](handler &CGH) {
      CGH.parallel_for<class UTNDRangeInvalidZeroLocal>(
          nd_range<1>{range<1>{8}, range<1>{0}}, [=](nd_item<1>) {});
    });
  } catch (const sycl::exception &E) {
    Thrown = true;
    EXPECT_EQ(E.code(), sycl::errc::nd_range);
  }

  EXPECT_TRUE(Thrown);
}
