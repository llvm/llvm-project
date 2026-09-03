//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <mock/helpers.hpp>

#include <sycl/__impl/device.hpp>
#include <sycl/__impl/queue.hpp>
#include <sycl/__impl/usm_functions.hpp>

#include <detail/device_impl.hpp>
#include <detail/queue_impl.hpp>

#include <cstddef>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

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

  EXPECT_CALL(Mock.get(), olMemAlloc(OLDev, OL_ALLOC_TYPE_DEVICE, NumBytes, _))
      .Times(1);
  void *Ptr1 = malloc_device(NumBytes, Dev, Ctx);
  EXPECT_NE(Ptr1, nullptr);

  EXPECT_CALL(Mock.get(), olMemFree(Ptr1)).Times(1);
  free(Ptr1, Ctx);

  EXPECT_CALL(Mock.get(), olMemAllocAligned(OLDev, OL_ALLOC_TYPE_DEVICE,
                                            NumBytes, Alignment, _))
      .Times(1);
  void *Ptr2 = aligned_alloc_device(Alignment, NumBytes, Q);
  EXPECT_NE(Ptr2, nullptr);

  EXPECT_CALL(Mock.get(), olMemFree(Ptr2)).Times(1);
  free(Ptr2, Q);
}

TEST(USMFunctions, HostAllocation) {
  mock::MockWrapper Mock;
  queue Q;
  context Ctx = Q.get_context();
  ol_device_handle_t OLDev =
      detail::getSyclObjImpl(Q.get_device())->getOLHandle();

  EXPECT_CALL(Mock.get(), olMemAllocHost(OLDev, NumBytes, _)).Times(1);
  void *Ptr1 = malloc_host(NumBytes, Ctx);
  EXPECT_NE(Ptr1, nullptr);

  EXPECT_CALL(Mock.get(), olMemFree(Ptr1)).Times(1);
  free(Ptr1, Ctx);

  EXPECT_CALL(Mock.get(), olMemAllocAlignedHost(OLDev, NumBytes, Alignment, _))
      .Times(1);
  void *Ptr2 = aligned_alloc_host(Alignment, NumBytes, Ctx);
  EXPECT_NE(Ptr2, nullptr);

  EXPECT_CALL(Mock.get(), olMemFree(Ptr2)).Times(1);
  free(Ptr2, Ctx);
}

TEST(USMFunctions, SharedAllocation) {
  mock::MockWrapper Mock;
  queue Q;
  device Dev = Q.get_device();
  context Ctx = Q.get_context();
  ol_device_handle_t OLDev = detail::getSyclObjImpl(Dev)->getOLHandle();

  EXPECT_CALL(Mock.get(), olMemAlloc(OLDev, OL_ALLOC_TYPE_MANAGED, NumBytes, _))
      .Times(1);
  void *Ptr1 = malloc_shared(NumBytes, Dev, Ctx);
  EXPECT_NE(Ptr1, nullptr);

  EXPECT_CALL(Mock.get(), olMemFree(Ptr1)).Times(1);
  free(Ptr1, Ctx);

  EXPECT_CALL(Mock.get(), olMemAllocAligned(OLDev, OL_ALLOC_TYPE_MANAGED,
                                            NumBytes, Alignment, _))
      .Times(1);
  void *Ptr2 = aligned_alloc_shared(Alignment, NumBytes, Q);
  EXPECT_NE(Ptr2, nullptr);

  EXPECT_CALL(Mock.get(), olMemFree(Ptr2)).Times(1);
  free(Ptr2, Q);
}

TEST(USMFunctions, ZeroByteAllocation) {
  mock::MockWrapper Mock;
  queue Q;
  device Dev = Q.get_device();
  context Ctx = Q.get_context();

  EXPECT_CALL(Mock.get(), olMemAlloc(_, _, _, _)).Times(0);
  EXPECT_CALL(Mock.get(), olMemAllocHost(_, _, _)).Times(0);
  EXPECT_CALL(Mock.get(), olMemAllocAligned(_, _, _, _, _)).Times(0);
  EXPECT_CALL(Mock.get(), olMemAllocAlignedHost(_, _, _, _)).Times(0);

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

  EXPECT_CALL(Mock.get(), olMemAllocAligned(OLDev, OL_ALLOC_TYPE_DEVICE,
                                            NumBytes, NonPowerOf2Alignment, _))
      .Times(1)
      .WillOnce(Return(mock::getMockLiboffload().makeEmptyStrError(
          OL_ERRC_INVALID_ARGUMENT)));
  EXPECT_EQ(aligned_alloc_device(NonPowerOf2Alignment, NumBytes, Dev, Ctx),
            nullptr);

  EXPECT_CALL(Mock.get(),
              olMemAllocAlignedHost(OLDev, NumBytes, NonPowerOf2Alignment, _))
      .Times(1)
      .WillOnce(Return(mock::getMockLiboffload().makeEmptyStrError(
          OL_ERRC_INVALID_ARGUMENT)));
  EXPECT_EQ(aligned_alloc_host(NonPowerOf2Alignment, NumBytes, Ctx), nullptr);

  EXPECT_CALL(Mock.get(), olMemAllocAligned(OLDev, OL_ALLOC_TYPE_MANAGED,
                                            NumBytes, NonPowerOf2Alignment, _))
      .Times(1)
      .WillOnce(Return(mock::getMockLiboffload().makeEmptyStrError(
          OL_ERRC_INVALID_ARGUMENT)));
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

  EXPECT_CALL(Mock.get(), olMemAlloc(OLDev, OL_ALLOC_TYPE_DEVICE, NumBytes, _))
      .Times(1);
  void *Ptr1 = aligned_alloc_device(ZeroAlignment, NumBytes, Dev, Ctx);
  EXPECT_NE(Ptr1, nullptr);
  EXPECT_CALL(Mock.get(), olMemFree(Ptr1)).Times(1);
  free(Ptr1, Ctx);

  EXPECT_CALL(Mock.get(), olMemAllocHost(OLDev, NumBytes, _)).Times(1);
  void *Ptr2 = aligned_alloc_host(ZeroAlignment, NumBytes, Ctx);
  EXPECT_NE(Ptr2, nullptr);
  EXPECT_CALL(Mock.get(), olMemFree(Ptr2)).Times(1);
  free(Ptr2, Ctx);

  EXPECT_CALL(Mock.get(), olMemAlloc(OLDev, OL_ALLOC_TYPE_MANAGED, NumBytes, _))
      .Times(1);
  void *Ptr3 = aligned_alloc_shared(ZeroAlignment, NumBytes, Dev, Ctx);
  EXPECT_NE(Ptr3, nullptr);
  EXPECT_CALL(Mock.get(), olMemFree(Ptr3)).Times(1);
  free(Ptr3, Ctx);
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

  EXPECT_CALL(Mock.get(), olMemAllocAligned(OLDev, OL_ALLOC_TYPE_DEVICE,
                                            sizeof(Over), alignof(Over), _))
      .Times(1);
  Over *P1 = aligned_alloc_device<Over>(1, 1, Dev, Ctx);
  EXPECT_NE(P1, nullptr);
  EXPECT_CALL(Mock.get(), olMemFree(P1)).Times(1);
  free(P1, Ctx);

  EXPECT_CALL(Mock.get(), olMemAllocAligned(OLDev, OL_ALLOC_TYPE_DEVICE,
                                            sizeof(Over), alignof(Over), _))
      .Times(1);
  Over *P2 = malloc_device<Over>(1, Dev, Ctx);
  EXPECT_NE(P2, nullptr);
  EXPECT_CALL(Mock.get(), olMemFree(P2)).Times(1);
  free(P2, Ctx);

  EXPECT_CALL(Mock.get(),
              olMemAllocAlignedHost(OLDev, sizeof(Over), alignof(Over), _))
      .Times(1);
  Over *P3 = aligned_alloc_host<Over>(1, 1, Ctx);
  EXPECT_NE(P3, nullptr);
  EXPECT_CALL(Mock.get(), olMemFree(P3)).Times(1);
  free(P3, Ctx);

  EXPECT_CALL(Mock.get(),
              olMemAllocAlignedHost(OLDev, sizeof(Over), alignof(Over), _))
      .Times(1);
  Over *P4 = malloc_host<Over>(1, Ctx);
  EXPECT_NE(P4, nullptr);
  EXPECT_CALL(Mock.get(), olMemFree(P4)).Times(1);
  free(P4, Ctx);

  EXPECT_CALL(Mock.get(), olMemAllocAligned(OLDev, OL_ALLOC_TYPE_MANAGED,
                                            sizeof(Over), alignof(Over), _))
      .Times(1);
  Over *P5 = aligned_alloc_shared<Over>(1, 1, Dev, Ctx);
  EXPECT_NE(P5, nullptr);
  EXPECT_CALL(Mock.get(), olMemFree(P5)).Times(1);
  free(P5, Ctx);

  EXPECT_CALL(Mock.get(), olMemAllocAligned(OLDev, OL_ALLOC_TYPE_MANAGED,
                                            sizeof(Over), alignof(Over), _))
      .Times(1);
  Over *P6 = malloc_shared<Over>(1, Dev, Ctx);
  EXPECT_NE(P6, nullptr);
  EXPECT_CALL(Mock.get(), olMemFree(P6)).Times(1);
  free(P6, Ctx);
}
