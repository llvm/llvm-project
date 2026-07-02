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

  EXPECT_CALL(Mock.get(), olGetMemInfo(Src, OL_MEM_INFO_DEVICE,
                                       sizeof(ol_device_handle_t), _))
      .WillRepeatedly([&](const void *Ptr, ol_mem_info_t PropName,
                          size_t PropSize, void *PropValue) -> ol_result_t {
        EXPECT_EQ(Ptr, static_cast<const void *>(Src));
        std::ignore = PropName;
        std::ignore = PropSize;
        *(static_cast<ol_device_handle_t *>(PropValue)) = OLDev;
        return OL_SUCCESS;
      });

  EXPECT_CALL(Mock.get(), olGetMemInfo(Dst, OL_MEM_INFO_DEVICE,
                                       sizeof(ol_device_handle_t), _))
      .WillRepeatedly([&](const void *Ptr, ol_mem_info_t PropName,
                          size_t PropSize, void *PropValue) -> ol_result_t {
        EXPECT_EQ(Ptr, static_cast<const void *>(Dst));
        std::ignore = PropName;
        std::ignore = PropSize;
        *(static_cast<ol_device_handle_t *>(PropValue)) = OLDev;
        return OL_SUCCESS;
      });

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

  EXPECT_CALL(Mock.get(), olGetMemInfo(_, OL_MEM_INFO_DEVICE,
                                       sizeof(ol_device_handle_t), _))
      .Times(4)
      .WillRepeatedly([&](const void *Ptr, ol_mem_info_t PropName,
                          size_t PropSize, void *PropValue) -> ol_result_t {
        EXPECT_TRUE(Ptr == static_cast<const void *>(SrcA) ||
                    Ptr == static_cast<const void *>(Mid) ||
                    Ptr == static_cast<const void *>(Dst));
        std::ignore = PropName;
        std::ignore = PropSize;
        *(static_cast<ol_device_handle_t *>(PropValue)) = OLDev;
        return OL_SUCCESS;
      });

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
