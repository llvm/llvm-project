#include "test_helpers.hpp"
#include <mock/helpers.hpp>

#include <detail/device_impl.hpp>

#include <sycl/__impl/device.hpp>
#include <sycl/__impl/queue.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace sycl;
using namespace ::testing;

TEST(Handler, MemcpyViaSubmit) {
  constexpr int NumBytes = 64;

  mock::MockWrapper Mock;
  queue Q;

  int Src[NumBytes / sizeof(int)] = {};
  int Dst[NumBytes / sizeof(int)] = {};

  ol_device_handle_t OLDev =
      detail::getSyclObjImpl(Q.get_device())->getOLHandle();

  sycl::unittests::expectDeviceMemoryInfo(Mock, {Src, Dst}, OLDev, 2);

  EXPECT_CALL(Mock.get(), olMemcpy(_, Dst, OLDev, Src, OLDev, NumBytes))
      .Times(1);
  EXPECT_CALL(Mock.get(), olCreateEvent(_, _, _)).Times(1);

  auto E = Q.submit([&](handler &CGH) { CGH.memcpy(Dst, Src, NumBytes); });

  EXPECT_CALL(Mock.get(), olSyncEvent(_)).Times(1);
  E.wait();
}

TEST(Handler, DependsOnWithMemcpy) {
  constexpr int NumBytes = 32;

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
  EXPECT_CALL(Mock.get(), olCreateEvent(_, _, _)).Times(2);

  auto First = Q.memcpy(Mid, SrcA, NumBytes);

  EXPECT_CALL(Mock.get(), olWaitEvents(_, _, 1)).Times(1);
  auto Second = Q.submit([&](handler &CGH) {
    CGH.depends_on(First);
    CGH.memcpy(Dst, Mid, NumBytes);
  });

  EXPECT_CALL(Mock.get(), olSyncEvent(_)).Times(1);
  Second.wait();
}
