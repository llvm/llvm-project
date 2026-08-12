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

TEST(USMFunctions, DeviceAllocation) {
  constexpr size_t NumBytes = 1024;
  constexpr size_t Alignment = 256;
  constexpr size_t DefaultAlign = alignof(std::max_align_t);

  mock::MockWrapper Mock;
  queue Q;
  device Dev = Q.get_device();
  context Ctx = Q.get_context();
  ol_device_handle_t OLDev = detail::getSyclObjImpl(Dev)->getOLHandle();

  // 1. Test malloc_device
  void *DummyPtr1 = mock::createDummyHandle<void *>();
  EXPECT_CALL(Mock.get(), olMemAllocAligned(OLDev, OL_ALLOC_TYPE_DEVICE,
                                            NumBytes, DefaultAlign, _))
      .Times(1)
      .WillOnce([&](ol_device_handle_t Device, ol_alloc_type_t AllocType,
                    size_t Size, size_t Alignment,
                    void **OutPtr) -> ol_result_t {
        *OutPtr = DummyPtr1;
        return OL_SUCCESS;
      });

  void *Ptr1 = malloc_device(NumBytes, Dev, Ctx);
  EXPECT_EQ(Ptr1, DummyPtr1);

  EXPECT_CALL(Mock.get(), olMemFree(Ptr1))
      .Times(1)
      .WillOnce(Return(ol_result_t(OL_SUCCESS)));
  free(Ptr1, Ctx);

  // 2. Test aligned_alloc_device
  void *DummyPtr2 = mock::createDummyHandle<void *>();
  EXPECT_CALL(Mock.get(), olMemAllocAligned(OLDev, OL_ALLOC_TYPE_DEVICE,
                                            NumBytes, Alignment, _))
      .Times(1)
      .WillOnce([&](ol_device_handle_t Device, ol_alloc_type_t AllocType,
                    size_t Size, size_t Alignment,
                    void **OutPtr) -> ol_result_t {
        *OutPtr = DummyPtr2;
        return OL_SUCCESS;
      });

  void *Ptr2 = aligned_alloc_device(Alignment, NumBytes, Q);
  EXPECT_EQ(Ptr2, DummyPtr2);

  EXPECT_CALL(Mock.get(), olMemFree(Ptr2))
      .Times(1)
      .WillOnce(Return(ol_result_t(OL_SUCCESS)));
  free(Ptr2, Q);
}

TEST(USMFunctions, HostAllocation) {
  constexpr size_t NumBytes = 512;
  constexpr size_t Alignment = 64;
  void *DummyPtr = mock::createDummyHandle<void *>();

  mock::MockWrapper Mock;
  queue Q;
  context Ctx = Q.get_context();
  ol_device_handle_t OLDev =
      detail::getSyclObjImpl(Q.get_device())->getOLHandle();

  EXPECT_CALL(Mock.get(), olMemAllocAlignedHost(OLDev, NumBytes, Alignment, _))
      .Times(1)
      .WillOnce([&](ol_device_handle_t Device, size_t Size, size_t Alignment,
                    void **OutPtr) -> ol_result_t {
        *OutPtr = DummyPtr;
        return OL_SUCCESS;
      });

  void *Ptr = aligned_alloc_host(Alignment, NumBytes, Ctx);
  EXPECT_EQ(Ptr, DummyPtr);

  EXPECT_CALL(Mock.get(), olMemFree(Ptr))
      .Times(1)
      .WillOnce(Return(ol_result_t(OL_SUCCESS)));
  free(Ptr, Ctx);
}

TEST(USMFunctions, SharedAllocation) {
  constexpr size_t NumBytes = 1024;
  constexpr size_t Alignment = 128;
  constexpr size_t DefaultAlign = alignof(std::max_align_t);

  mock::MockWrapper Mock;
  queue Q;
  device Dev = Q.get_device();
  context Ctx = Q.get_context();
  ol_device_handle_t OLDev = detail::getSyclObjImpl(Dev)->getOLHandle();

  // 1. Test malloc_shared
  void *DummyPtr1 = mock::createDummyHandle<void *>();
  EXPECT_CALL(Mock.get(), olMemAllocAligned(OLDev, OL_ALLOC_TYPE_MANAGED,
                                            NumBytes, DefaultAlign, _))
      .Times(1)
      .WillOnce([&](ol_device_handle_t Device, ol_alloc_type_t AllocType,
                    size_t Size, size_t Alignment,
                    void **OutPtr) -> ol_result_t {
        *OutPtr = DummyPtr1;
        return OL_SUCCESS;
      });

  void *Ptr1 = malloc_shared(NumBytes, Dev, Ctx);
  EXPECT_EQ(Ptr1, DummyPtr1);

  EXPECT_CALL(Mock.get(), olMemFree(Ptr1))
      .Times(1)
      .WillOnce(Return(ol_result_t(OL_SUCCESS)));
  free(Ptr1, Ctx);

  // 2. Test aligned_alloc_shared
  void *DummyPtr2 = mock::createDummyHandle<void *>();
  EXPECT_CALL(Mock.get(), olMemAllocAligned(OLDev, OL_ALLOC_TYPE_MANAGED,
                                            NumBytes, Alignment, _))
      .Times(1)
      .WillOnce([&](ol_device_handle_t Device, ol_alloc_type_t AllocType,
                    size_t Size, size_t Alignment,
                    void **OutPtr) -> ol_result_t {
        *OutPtr = DummyPtr2;
        return OL_SUCCESS;
      });

  void *Ptr2 = aligned_alloc_shared(Alignment, NumBytes, Q);
  EXPECT_EQ(Ptr2, DummyPtr2);

  EXPECT_CALL(Mock.get(), olMemFree(Ptr2))
      .Times(1)
      .WillOnce(Return(ol_result_t(OL_SUCCESS)));
  free(Ptr2, Q);
}
