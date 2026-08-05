#include <mock/helpers.hpp>

#include <sycl/__impl/device.hpp>
#include <sycl/__impl/queue.hpp>

#include <detail/device_impl.hpp>
#include <detail/queue_impl.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace sycl;
using namespace ::testing;

TEST(Queue, Memcpy) {
  constexpr int NumBytes = 32;
  constexpr int NMemcpies = 4;
  constexpr int NMemcpyAttempts = 5;

  mock::MockWrapper Mock;
  queue Q;

  bool IsSrcHostPtr = false;
  bool IsDstHostPtr = false;
  int *SrcPtr = reinterpret_cast<int *>(1);
  int *DstPtr = reinterpret_cast<int *>(2);
  ol_device_handle_t OLDev =
      detail::getSyclObjImpl(Q.get_device())->getOLHandle();

  EXPECT_CALL(Mock.get(), olGetMemInfo(_, OL_MEM_INFO_DEVICE,
                                       sizeof(ol_device_handle_t), _))
      .Times(NMemcpyAttempts * 2)
      .WillRepeatedly([&](const void *Ptr, ol_mem_info_t PropName,
                          size_t PropSize, void *PropValue) -> ol_result_t {
        EXPECT_TRUE(Ptr == SrcPtr || Ptr == DstPtr);
        bool IsHostPtr = Ptr == SrcPtr ? IsSrcHostPtr : IsDstHostPtr;
        if (IsHostPtr)
          return mock::getMockLiboffload().makeEmptyStrError(OL_ERRC_NOT_FOUND);
        *(static_cast<ol_device_handle_t *>(PropValue)) = OLDev;
        return OL_SUCCESS;
      });
  EXPECT_CALL(Mock.get(), olMemcpy(_, DstPtr, _, SrcPtr, _, NumBytes))
      .Times(NMemcpies)
      .WillRepeatedly([&](ol_queue_handle_t Queue, void *DstPtr,
                          ol_device_handle_t DstDevice, const void *SrcPtr,
                          ol_device_handle_t SrcDevice,
                          size_t Size) -> ol_result_t {
        EXPECT_NE(Queue, nullptr);
        ol_device_handle_t HostDevice =
            mock::getMockLiboffload().getHostOLDevice();
        EXPECT_EQ(DstDevice, IsDstHostPtr ? HostDevice : OLDev);
        EXPECT_EQ(SrcDevice, IsSrcHostPtr ? HostDevice : OLDev);
        return OL_SUCCESS;
      });

  EXPECT_CALL(Mock.get(), olCreateEvent(_, _, _)).Times(NMemcpies);

  event Event = Q.memcpy(DstPtr, SrcPtr, NumBytes);

  EXPECT_CALL(Mock.get(), olWaitEvents(_, _, 1));
  Q.memcpy(DstPtr, SrcPtr, NumBytes, Event);

  IsSrcHostPtr = true;
  Q.memcpy(DstPtr, SrcPtr, NumBytes);
  IsSrcHostPtr = false;
  IsDstHostPtr = true;
  Q.memcpy(DstPtr, SrcPtr, NumBytes);

  IsSrcHostPtr = true;
  try {
    Q.memcpy(DstPtr, SrcPtr, NumBytes);
  } catch (const exception &e) {
    EXPECT_EQ(e.code(), make_error_code(errc::feature_not_supported));
  }
}

TEST(Queue, MemcpyZeroBytes) {
  mock::MockWrapper Mock;
  queue Q;
  EXPECT_CALL(Mock.get(), olWaitEvents(_, _, 1)).Times(1);
  EXPECT_CALL(Mock.get(), olGetMemInfo(_, _, _, _)).Times(0);
  EXPECT_CALL(Mock.get(), olMemcpy(_, _, _, _, _, _)).Times(0);
  event Event = Q.memcpy(nullptr, nullptr, 0);
  Q.memcpy(nullptr, nullptr, 0, Event);
}
