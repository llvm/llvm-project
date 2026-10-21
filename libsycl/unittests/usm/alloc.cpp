//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <common/unittests_helper.hpp>
#include <mock/helpers.hpp>

#include <sycl/__impl/device.hpp>
#include <sycl/__impl/platform.hpp>
#include <sycl/__impl/queue.hpp>
#include <sycl/__impl/usm_functions.hpp>

#include <detail/device_impl.hpp>
#include <detail/queue_impl.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <vector>

using namespace sycl;
using namespace ::testing;

constexpr size_t NumBytes = 1024;
constexpr size_t Alignment = 256;

TEST(USMFunctions, DeviceAllocation) {
  mock::MockWrapper Mock;
  queue Q;
  device Dev = Q.get_device();
  context Ctx = Q.get_context();
  ol_device_handle_t OLDev = detail::getSyclObjImpl(Dev)->getOLHandle();

  EXPECT_CALL(Mock.get(), olMemAllocAligned(_, _, _, _, _, _)).Times(0);
  EXPECT_CALL(Mock.get(),
              olMemAlloc(_, OLDev, OL_ALLOC_TYPE_DEVICE, NumBytes, _))
      .Times(1);
  void *Ptr1 = malloc_device(NumBytes, Dev, Ctx);
  EXPECT_NE(Ptr1, nullptr);

  EXPECT_CALL(Mock.get(), olMemFree(_, Ptr1)).Times(1);
  free(Ptr1, Ctx);

  EXPECT_CALL(Mock.get(), olMemAlloc(_, _, _, _, _)).Times(0);
  EXPECT_CALL(Mock.get(), olMemAllocAligned(_, OLDev, OL_ALLOC_TYPE_DEVICE,
                                            NumBytes, Alignment, _))
      .Times(1);
  void *Ptr2 = aligned_alloc_device(Alignment, NumBytes, Q);
  EXPECT_NE(Ptr2, nullptr);

  EXPECT_CALL(Mock.get(), olMemFree(_, Ptr2)).Times(1);
  free(Ptr2, Q);
}

TEST(USMFunctions, HostAllocation) {
  mock::MockWrapper Mock;
  queue Q;
  context Ctx = Q.get_context();
  ol_device_handle_t OLDev =
      detail::getSyclObjImpl(Q.get_device())->getOLHandle();

  EXPECT_CALL(Mock.get(), olMemAllocAlignedHost(_, _, _, _, _)).Times(0);
  EXPECT_CALL(Mock.get(), olMemAllocHost(_, OLDev, NumBytes, _)).Times(1);
  void *Ptr1 = malloc_host(NumBytes, Ctx);
  EXPECT_NE(Ptr1, nullptr);

  EXPECT_CALL(Mock.get(), olMemFree(_, Ptr1)).Times(1);
  free(Ptr1, Ctx);

  EXPECT_CALL(Mock.get(), olMemAllocHost(_, _, _, _)).Times(0);
  EXPECT_CALL(Mock.get(),
              olMemAllocAlignedHost(_, OLDev, NumBytes, Alignment, _))
      .Times(1);
  void *Ptr2 = aligned_alloc_host(Alignment, NumBytes, Ctx);
  EXPECT_NE(Ptr2, nullptr);

  EXPECT_CALL(Mock.get(), olMemFree(_, Ptr2)).Times(1);
  free(Ptr2, Ctx);
}

TEST(USMFunctions, SharedAllocation) {
  mock::MockWrapper Mock;
  queue Q;
  device Dev = Q.get_device();
  context Ctx = Q.get_context();
  ol_device_handle_t OLDev = detail::getSyclObjImpl(Dev)->getOLHandle();

  EXPECT_CALL(Mock.get(), olMemAllocAligned(_, _, _, _, _, _)).Times(0);
  EXPECT_CALL(Mock.get(),
              olMemAlloc(_, OLDev, OL_ALLOC_TYPE_MANAGED, NumBytes, _))
      .Times(1);
  void *Ptr1 = malloc_shared(NumBytes, Dev, Ctx);
  EXPECT_NE(Ptr1, nullptr);

  EXPECT_CALL(Mock.get(), olMemFree(_, Ptr1)).Times(1);
  free(Ptr1, Ctx);

  EXPECT_CALL(Mock.get(), olMemAlloc(_, _, _, _, _)).Times(0);
  EXPECT_CALL(Mock.get(), olMemAllocAligned(_, OLDev, OL_ALLOC_TYPE_MANAGED,
                                            NumBytes, Alignment, _))
      .Times(1);
  void *Ptr2 = aligned_alloc_shared(Alignment, NumBytes, Q);
  EXPECT_NE(Ptr2, nullptr);

  EXPECT_CALL(Mock.get(), olMemFree(_, Ptr2)).Times(1);
  free(Ptr2, Q);
}

TEST(USMFunctions, ZeroByteAllocation) {
  mock::MockWrapper Mock;
  queue Q;
  device Dev = Q.get_device();
  context Ctx = Q.get_context();

  EXPECT_CALL(Mock.get(), olMemAlloc(_, _, _, _, _)).Times(0);
  EXPECT_CALL(Mock.get(), olMemAllocHost(_, _, _, _)).Times(0);
  EXPECT_CALL(Mock.get(), olMemAllocAligned(_, _, _, _, _, _)).Times(0);
  EXPECT_CALL(Mock.get(), olMemAllocAlignedHost(_, _, _, _, _)).Times(0);

  EXPECT_EQ(malloc_device(0, Dev, Ctx), nullptr);
  EXPECT_EQ(malloc_shared(0, Dev, Ctx), nullptr);
  EXPECT_EQ(malloc_host(0, Ctx), nullptr);
}

TEST(USMFunctions, InvalidAlignment) {
  mock::MockWrapper Mock;
  queue Q;
  device Dev = Q.get_device();
  context Ctx = Q.get_context();
  ol_device_handle_t OLDev = detail::getSyclObjImpl(Dev)->getOLHandle();

  constexpr size_t NonPowerOf2Alignment = 3;

  EXPECT_CALL(Mock.get(), olMemAlloc(_, _, _, _, _)).Times(0);
  EXPECT_CALL(Mock.get(), olMemAllocHost(_, _, _, _)).Times(0);

  EXPECT_CALL(Mock.get(), olMemAllocAligned(_, OLDev, OL_ALLOC_TYPE_DEVICE,
                                            NumBytes, NonPowerOf2Alignment, _))
      .Times(1);
  EXPECT_EQ(aligned_alloc_device(NonPowerOf2Alignment, NumBytes, Dev, Ctx),
            nullptr);

  EXPECT_CALL(Mock.get(), olMemAllocAlignedHost(_, OLDev, NumBytes,
                                                NonPowerOf2Alignment, _))
      .Times(1);
  EXPECT_EQ(aligned_alloc_host(NonPowerOf2Alignment, NumBytes, Ctx), nullptr);

  EXPECT_CALL(Mock.get(), olMemAllocAligned(_, OLDev, OL_ALLOC_TYPE_MANAGED,
                                            NumBytes, NonPowerOf2Alignment, _))
      .Times(1);
  EXPECT_EQ(aligned_alloc_shared(NonPowerOf2Alignment, NumBytes, Dev, Ctx),
            nullptr);
}

TEST(USMFunctions, ZeroAlignmentSucceeds) {
  mock::MockWrapper Mock;
  queue Q;
  device Dev = Q.get_device();
  context Ctx = Q.get_context();
  ol_device_handle_t OLDev = detail::getSyclObjImpl(Dev)->getOLHandle();

  constexpr size_t ZeroAlignment = 0;

  EXPECT_CALL(Mock.get(), olMemAllocAligned(_, _, _, _, _, _)).Times(0);
  EXPECT_CALL(Mock.get(), olMemAllocAlignedHost(_, _, _, _, _)).Times(0);

  EXPECT_CALL(Mock.get(),
              olMemAlloc(_, OLDev, OL_ALLOC_TYPE_DEVICE, NumBytes, _))
      .Times(1);
  void *Ptr1 = aligned_alloc_device(ZeroAlignment, NumBytes, Dev, Ctx);
  EXPECT_NE(Ptr1, nullptr);
  EXPECT_CALL(Mock.get(), olMemFree(_, Ptr1)).Times(1);
  free(Ptr1, Ctx);

  EXPECT_CALL(Mock.get(), olMemAllocHost(_, OLDev, NumBytes, _)).Times(1);
  void *Ptr2 = aligned_alloc_host(ZeroAlignment, NumBytes, Ctx);
  EXPECT_NE(Ptr2, nullptr);
  EXPECT_CALL(Mock.get(), olMemFree(_, Ptr2)).Times(1);
  free(Ptr2, Ctx);

  EXPECT_CALL(Mock.get(),
              olMemAlloc(_, OLDev, OL_ALLOC_TYPE_MANAGED, NumBytes, _))
      .Times(1);
  void *Ptr3 = aligned_alloc_shared(ZeroAlignment, NumBytes, Dev, Ctx);
  EXPECT_NE(Ptr3, nullptr);
  EXPECT_CALL(Mock.get(), olMemFree(_, Ptr3)).Times(1);
  free(Ptr3, Ctx);
}

TEST(USMFunctions, UnknownAllocationKindThrows) {
  mock::MockWrapper Mock;
  queue Q;
  device Dev = Q.get_device();
  context Ctx = Q.get_context();

  EXPECT_CALL(Mock.get(), olMemAlloc(_, _, _, _, _)).Times(0);
  EXPECT_CALL(Mock.get(), olMemAllocHost(_, _, _, _)).Times(0);

  try {
    sycl::malloc(NumBytes, Dev, Ctx, usm::alloc::unknown);
    FAIL() << "Expected sycl::exception";
  } catch (const sycl::exception &E) {
    EXPECT_EQ(E.code(), make_error_code(errc::invalid));
    EXPECT_TRUE(E.has_context());
    EXPECT_EQ(E.get_context(), Ctx);
  }
}

namespace {

// The default mock exposes a single device, so device enumeration has to be
// mocked to get a context that doesn't contain a device.
class USMTwoDevicesTest : public Test {
protected:
  void SetUp() override {
    Platform = mock::createDummyHandle<ol_platform_handle_t>();
    for (ol_device_handle_t &Device : Devices)
      Device = mock::createDummyHandleWithData<ol_device_handle_t>(
          reinterpret_cast<unsigned char *>(&Platform), sizeof(Platform));

    EXPECT_CALL(Helper.Mock.get(), olIterateDevices(_, _))
        .WillRepeatedly([this](ol_device_iterate_cb_t Callback,
                               void *UserData) -> ol_result_t {
          for (ol_device_handle_t Device : Devices)
            std::ignore = Callback(Device, UserData);
          return OL_SUCCESS;
        });
  }

  void TearDown() override {
    mock::releaseDummyHandles(Devices[0], Devices[1], Platform);
  }

  unittests::UnittestsHelper Helper;
  ol_platform_handle_t Platform{};
  std::array<ol_device_handle_t, 2> Devices{};
};

} // namespace

TEST_F(USMTwoDevicesTest, DeviceNotInContextThrows) {
  std::vector<platform> Platforms = platform::get_platforms();
  ASSERT_EQ(Platforms.size(), 1u);

  std::vector<device> PlatformDevices = Platforms[0].get_devices();
  ASSERT_EQ(PlatformDevices.size(), 2u);

  context Ctx(PlatformDevices[0]);

  EXPECT_CALL(Helper.Mock.get(), olMemAlloc(_, _, _, _, _)).Times(0);
  EXPECT_CALL(Helper.Mock.get(), olMemAllocAligned(_, _, _, _, _, _)).Times(0);

  try {
    malloc_device(NumBytes, PlatformDevices[1], Ctx);
    FAIL() << "Expected sycl::exception";
  } catch (const sycl::exception &E) {
    EXPECT_EQ(E.code(), make_error_code(errc::invalid));
    EXPECT_TRUE(E.has_context());
    EXPECT_EQ(E.get_context(), Ctx);
  }
}

struct alignas(64) Over {
  char c;
};

TEST(USMFunctions, TemplatedAlignment) {
  mock::MockWrapper Mock;
  queue Q;
  device Dev = Q.get_device();
  context Ctx = Q.get_context();
  ol_device_handle_t OLDev = detail::getSyclObjImpl(Dev)->getOLHandle();

  EXPECT_CALL(Mock.get(), olMemAlloc(_, _, _, _, _)).Times(0);
  EXPECT_CALL(Mock.get(), olMemAllocHost(_, _, _, _)).Times(0);

  EXPECT_CALL(Mock.get(), olMemAllocAligned(_, OLDev, OL_ALLOC_TYPE_DEVICE,
                                            sizeof(Over), alignof(Over), _))
      .Times(1);
  Over *P1 = aligned_alloc_device<Over>(1, 1, Dev, Ctx);
  EXPECT_NE(P1, nullptr);
  EXPECT_CALL(Mock.get(), olMemFree(_, P1)).Times(1);
  free(P1, Ctx);

  EXPECT_CALL(Mock.get(), olMemAllocAligned(_, OLDev, OL_ALLOC_TYPE_DEVICE,
                                            sizeof(Over), alignof(Over), _))
      .Times(1);
  Over *P2 = malloc_device<Over>(1, Dev, Ctx);
  EXPECT_NE(P2, nullptr);
  EXPECT_CALL(Mock.get(), olMemFree(_, P2)).Times(1);
  free(P2, Ctx);

  EXPECT_CALL(Mock.get(),
              olMemAllocAlignedHost(_, OLDev, sizeof(Over), alignof(Over), _))
      .Times(1);
  Over *P3 = aligned_alloc_host<Over>(1, 1, Ctx);
  EXPECT_NE(P3, nullptr);
  EXPECT_CALL(Mock.get(), olMemFree(_, P3)).Times(1);
  free(P3, Ctx);

  EXPECT_CALL(Mock.get(),
              olMemAllocAlignedHost(_, OLDev, sizeof(Over), alignof(Over), _))
      .Times(1);
  Over *P4 = malloc_host<Over>(1, Ctx);
  EXPECT_NE(P4, nullptr);
  EXPECT_CALL(Mock.get(), olMemFree(_, P4)).Times(1);
  free(P4, Ctx);

  EXPECT_CALL(Mock.get(), olMemAllocAligned(_, OLDev, OL_ALLOC_TYPE_MANAGED,
                                            sizeof(Over), alignof(Over), _))
      .Times(1);
  Over *P5 = aligned_alloc_shared<Over>(1, 1, Dev, Ctx);
  EXPECT_NE(P5, nullptr);
  EXPECT_CALL(Mock.get(), olMemFree(_, P5)).Times(1);
  free(P5, Ctx);

  EXPECT_CALL(Mock.get(), olMemAllocAligned(_, OLDev, OL_ALLOC_TYPE_MANAGED,
                                            sizeof(Over), alignof(Over), _))
      .Times(1);
  Over *P6 = malloc_shared<Over>(1, Dev, Ctx);
  EXPECT_NE(P6, nullptr);
  EXPECT_CALL(Mock.get(), olMemFree(_, P6)).Times(1);
  free(P6, Ctx);
}
