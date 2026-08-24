//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <common/unittests_helper.hpp>

#include <detail/platform_impl.hpp>
#include <sycl/__impl/detail/obj_utils.hpp>
#include <sycl/sycl.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
// #include <tuple>

using namespace sycl;
using namespace ::testing;

TEST(PlatformTest, APIGetPlatformsDefaultMock) {
  unittests::UnittestsHelper Helper;

  auto Platforms = sycl::platform::get_platforms();
  ASSERT_EQ(Platforms.size(), 1u);
  EXPECT_EQ(Platforms[0].get_backend(), sycl::backend::level_zero);

  auto Devices = Platforms[0].get_devices();
  ASSERT_EQ(Devices.size(), 1u);
  EXPECT_EQ(Devices[0].get_backend(), sycl::backend::level_zero);

  EXPECT_FALSE(Devices[0].is_cpu());
  EXPECT_FALSE(Devices[0].is_accelerator());
  EXPECT_TRUE(Devices[0].is_gpu());

  EXPECT_EQ(Devices[0].get_platform(), Platforms[0]);

#ifdef SYCL_KHR_DEFAULT_CONTEXT
  auto Context = Platforms[0].khr_get_default_context();
  EXPECT_EQ(Context.get_platform(), Platforms[0]);

  auto CtxDevices = Context.get_devices();
  ASSERT_EQ(CtxDevices.size(), 1u);
  ASSERT_EQ(CtxDevices[0], Devices[0]);
#endif
}

namespace {

class PlatformContextGroupTest : public Test {
protected:
  void SetUp() override {
    Platform = mock::createDummyHandle<ol_platform_handle_t>();
    for (ol_device_handle_t &Device : Devices) {
      Device = mock::createDummyHandleWithData<ol_device_handle_t>(
          reinterpret_cast<unsigned char *>(&Platform), sizeof(Platform));
    }

    EXPECT_CALL(Helper.Mock.get(), olIterateDevices(_, _))
        .WillRepeatedly([this](ol_device_iterate_cb_t Callback,
                               void *UserData) -> ol_result_t {
          for (ol_device_handle_t Device : Devices)
            std::ignore = Callback(Device, UserData);
          return OL_SUCCESS;
        });

    ON_CALL(Helper.Mock.get(),
            olGetDeviceInfo(_, OL_DEVICE_INFO_CONTEXT_GROUP_INDEX, _, _))
        .WillByDefault([this](ol_device_handle_t Device,
                              ol_device_info_t /*PropName*/, size_t PropSize,
                              void *PropValue) -> ol_result_t {
          EXPECT_EQ(PropSize, sizeof(uint32_t));
          if (FailContextGroupQuery)
            return Helper.Mock.get().makeEmptyStrError(OL_ERRC_UNIMPLEMENTED);

          *static_cast<uint32_t *>(PropValue) = getContextGroup(Device);
          return OL_SUCCESS;
        });
  }

  void TearDown() override {
    detail::getPlatformCache().clear();
    detail::getOffloadTopologies() = {};
    mock::releaseDummyHandles(Devices[0], Devices[1], Devices[2], Platform);
  }

  uint32_t getContextGroup(ol_device_handle_t Device) const {
    if (Device == Devices[0] || Device == Devices[2])
      return 0;
    if (Device == Devices[1])
      return 1;
    ADD_FAILURE() << "Unexpected device";
    return 0;
  }

  unittests::UnittestsHelper Helper;
  ol_platform_handle_t Platform{};
  std::array<ol_device_handle_t, 3> Devices{};
  bool FailContextGroupQuery = false;
};

TEST_F(PlatformContextGroupTest, CreatesPlatformForEachContextGroup) {
  EXPECT_CALL(Helper.Mock.get(), olCreateContext(_, _, _))
      .Times(2)
      .WillRepeatedly([this](size_t NumDevices,
                             ol_device_handle_t *ContextDevices,
                             ol_context_handle_t *Context) -> ol_result_t {
        EXPECT_GT(NumDevices, 0u);
        if (NumDevices == 0)
          return Helper.Mock.get().makeEmptyStrError(OL_ERRC_INVALID_SIZE);
        const uint32_t ContextGroup = getContextGroup(ContextDevices[0]);
        for (size_t I = 1; I < NumDevices; ++I)
          EXPECT_EQ(getContextGroup(ContextDevices[I]), ContextGroup);

        *Context = mock::createDummyHandleWithData<ol_context_handle_t>(
            reinterpret_cast<unsigned char *>(&ContextDevices[0]),
            sizeof(ContextDevices[0]));
        return OL_SUCCESS;
      });

  auto Platforms = sycl::platform::get_platforms();
  ASSERT_EQ(Platforms.size(), 2u);
  EXPECT_NE(Platforms[0], Platforms[1]);

  auto Platform0Devices = Platforms[0].get_devices();
  ASSERT_EQ(Platform0Devices.size(), 2u);
  EXPECT_EQ(detail::getSyclObjImpl(Platform0Devices[0])->getOLHandle(),
            Devices[0]);
  EXPECT_EQ(detail::getSyclObjImpl(Platform0Devices[1])->getOLHandle(),
            Devices[2]);

  auto Platform1Devices = Platforms[1].get_devices();
  ASSERT_EQ(Platform1Devices.size(), 1u);
  EXPECT_EQ(detail::getSyclObjImpl(Platform1Devices[0])->getOLHandle(),
            Devices[1]);

  EXPECT_EQ(Platform0Devices[0].get_platform(), Platforms[0]);
  EXPECT_EQ(Platform0Devices[1].get_platform(), Platforms[0]);
  EXPECT_EQ(Platform1Devices[0].get_platform(), Platforms[1]);

  EXPECT_EQ(detail::getSyclObjImpl(Platforms[0])->getOLHandleRef(), Platform);
  EXPECT_EQ(detail::getSyclObjImpl(Platforms[1])->getOLHandleRef(), Platform);

  context Context0 = Platforms[0].khr_get_default_context();
  context Context1 = Platforms[1].khr_get_default_context();
  EXPECT_EQ(Context0.get_platform(), Platforms[0]);
  EXPECT_EQ(Context1.get_platform(), Platforms[1]);
  EXPECT_EQ(Context0.get_devices(), Platform0Devices);
  EXPECT_EQ(Context1.get_devices(), Platform1Devices);

  queue Queue0{Platform0Devices[0]};
  queue Queue1{Platform1Devices[0]};
  EXPECT_EQ(Queue0.get_context(), Context0);
  EXPECT_EQ(Queue1.get_context(), Context1);
}

TEST_F(PlatformContextGroupTest, ContextGroupQueryFailureUsesDefaultGroup) {
  FailContextGroupQuery = true;

  EXPECT_CALL(Helper.Mock.get(), olCreateContext(3, _, _)).Times(1);

  auto Platforms = sycl::platform::get_platforms();
  ASSERT_EQ(Platforms.size(), 1u);

  auto PlatformDevices = Platforms[0].get_devices();
  ASSERT_EQ(PlatformDevices.size(), Devices.size());
  EXPECT_EQ(detail::getSyclObjImpl(PlatformDevices[0])->getOLHandle(),
            Devices[0]);
  EXPECT_EQ(detail::getSyclObjImpl(PlatformDevices[1])->getOLHandle(),
            Devices[1]);
  EXPECT_EQ(detail::getSyclObjImpl(PlatformDevices[2])->getOLHandle(),
            Devices[2]);
  EXPECT_EQ(Platforms[0].khr_get_default_context().get_devices(),
            PlatformDevices);
}

} // namespace
