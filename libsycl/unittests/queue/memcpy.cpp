#include <common/unittests_helper.hpp>
#include <mock/helpers.hpp>

#include <sycl/__impl/device.hpp>
#include <sycl/__impl/platform.hpp>
#include <sycl/__impl/queue.hpp>

#include <detail/device_impl.hpp>
#include <detail/queue_impl.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>

using namespace sycl;
using namespace ::testing;

TEST(Queue, Memcpy) {
  constexpr int NumBytes = 32;
  constexpr int NMemcpies = 5;

  mock::MockWrapper Mock;
  queue Q;

  bool IsSrcHostPtr = false;
  bool IsDstHostPtr = false;
  int *SrcPtr = reinterpret_cast<int *>(1);
  int *DstPtr = reinterpret_cast<int *>(2);
  ol_device_handle_t OLDev =
      detail::getSyclObjImpl(Q.get_device())->getOLHandle();

  EXPECT_CALL(Mock.get(), olGetMemInfo(_, _, OL_MEM_INFO_DEVICE,
                                       sizeof(ol_device_handle_t), _))
      .Times(NMemcpies * 2)
      .WillRepeatedly([&](ol_context_handle_t Context, const void *Ptr,
                          ol_mem_info_t PropName, size_t PropSize,
                          void *PropValue) -> ol_result_t {
        EXPECT_NE(Context, nullptr);
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
  Q.memcpy(DstPtr, SrcPtr, NumBytes);
}

TEST(Queue, MemcpyZeroBytes) {
  mock::MockWrapper Mock;
  queue Q;
  EXPECT_CALL(Mock.get(), olWaitEvents(_, _, 1)).Times(1);
  EXPECT_CALL(Mock.get(), olGetMemInfo(_, _, _, _, _)).Times(0);
  EXPECT_CALL(Mock.get(), olMemcpy(_, _, _, _, _, _)).Times(0);
  event Event = Q.memcpy(nullptr, nullptr, 0);
  Q.memcpy(nullptr, nullptr, 0, Event);
}

TEST(Queue, MemcpyNullptrThrows) {
  constexpr int NumBytes = 32;

  mock::MockWrapper Mock;
  queue Q;
  int Src = 0;

  EXPECT_CALL(Mock.get(), olGetMemInfo(_, _, _, _, _)).Times(0);
  EXPECT_CALL(Mock.get(), olMemcpy(_, _, _, _, _, _)).Times(0);

  try {
    Q.memcpy(nullptr, &Src, NumBytes);
    FAIL() << "Expected sycl::exception";
  } catch (const sycl::exception &E) {
    EXPECT_EQ(E.code(), make_error_code(errc::invalid));
    EXPECT_TRUE(E.has_context());
    EXPECT_EQ(E.get_context(), Q.get_context());
  }
}

namespace {

// The default mock exposes a single platform, so device enumeration has to be
// mocked to get events that belong to different platforms. Platforms are formed
// per liboffload driver id.
class QueueTwoPlatformsTest : public Test {
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

    ON_CALL(Helper.Mock.get(),
            olGetDeviceInfo(_, OL_DEVICE_INFO_DRIVER_ID, _, _))
        .WillByDefault([this](ol_device_handle_t Device,
                              ol_device_info_t /*PropName*/, size_t PropSize,
                              void *PropValue) -> ol_result_t {
          EXPECT_EQ(PropSize, sizeof(uint32_t));
          *static_cast<uint32_t *>(PropValue) = Device == Devices[0] ? 0u : 1u;
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

TEST_F(QueueTwoPlatformsTest, CrossPlatformDependencyThrows) {
  std::vector<platform> Platforms = platform::get_platforms();
  ASSERT_EQ(Platforms.size(), 2u);

  std::vector<device> Devices0 = Platforms[0].get_devices();
  std::vector<device> Devices1 = Platforms[1].get_devices();
  ASSERT_EQ(Devices0.size(), 1u);
  ASSERT_EQ(Devices1.size(), 1u);

  queue Q0(Devices0.front());
  queue Q1(Devices1.front());

  event OtherPlatformEvent = Q1.memcpy(nullptr, nullptr, 0);

  try {
    Q0.memcpy(nullptr, nullptr, 0, OtherPlatformEvent);
    FAIL() << "Expected sycl::exception";
  } catch (const sycl::exception &E) {
    EXPECT_EQ(E.code(), make_error_code(errc::feature_not_supported));
    EXPECT_TRUE(E.has_context());
    EXPECT_EQ(E.get_context(), Q0.get_context());
  }
}
