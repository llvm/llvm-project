//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <common/unittests_helper.hpp>

#include <detail/device_impl.hpp>
#include <sycl/__impl/detail/obj_utils.hpp>
#include <sycl/sycl.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <tuple>
#include <vector>

using namespace sycl;
using namespace ::testing;

namespace {

// An aspect and the liboffload device query that reports it.
struct AspectQuery {
  aspect Aspect;
  ol_device_info_t Info;
};

class DeviceHasAspectTest : public TestWithParam<AspectQuery> {
protected:
  // Device discovery queries other device information through the default
  // mock. Expectations added by the tests are newer, so they take precedence.
  void SetUp() override {
    EXPECT_CALL(Helper.Mock.get(), olGetDeviceInfo(_, _, _, _))
        .Times(AnyNumber());
  }

  // Makes the mocked liboffload answer the query of the tested aspect with
  // Supported, for Device or, if Device is null, for every device.
  void mockSupport(bool Supported, ol_device_handle_t Device = nullptr);

  unittests::UnittestsHelper Helper;
};

void DeviceHasAspectTest::mockSupport(bool Supported,
                                      ol_device_handle_t Device) {
  Matcher<ol_device_handle_t> DeviceMatcher = _;
  if (Device)
    DeviceMatcher = Eq(Device);
  EXPECT_CALL(Helper.Mock.get(),
              olGetDeviceInfo(DeviceMatcher, GetParam().Info, _, _))
      .WillRepeatedly([Supported](ol_device_handle_t, ol_device_info_t,
                                  size_t PropSize,
                                  void *PropValue) -> ol_result_t {
        EXPECT_EQ(PropSize, sizeof(bool));
        *static_cast<bool *>(PropValue) = Supported;
        return OL_SUCCESS;
      });
}

TEST_P(DeviceHasAspectTest, Supported) {
  mockSupport(true);

  platform Platform = platform::get_platforms().at(0);
  EXPECT_TRUE(Platform.get_devices().at(0).has(GetParam().Aspect));
  EXPECT_TRUE(Platform.has(GetParam().Aspect));
}

TEST_P(DeviceHasAspectTest, NotSupported) {
  mockSupport(false);

  platform Platform = platform::get_platforms().at(0);
  EXPECT_FALSE(Platform.get_devices().at(0).has(GetParam().Aspect));
  EXPECT_FALSE(Platform.has(GetParam().Aspect));
}

TEST_P(DeviceHasAspectTest, PlatformRequiresAllDevices) {
  ol_platform_handle_t OLPlatform =
      mock::createDummyHandle<ol_platform_handle_t>();
  ol_device_handle_t WithAspect =
      mock::createDummyHandleWithData<ol_device_handle_t>(
          reinterpret_cast<unsigned char *>(&OLPlatform), sizeof(OLPlatform));
  ol_device_handle_t WithoutAspect =
      mock::createDummyHandleWithData<ol_device_handle_t>(
          reinterpret_cast<unsigned char *>(&OLPlatform), sizeof(OLPlatform));
  EXPECT_CALL(Helper.Mock.get(), olIterateDevices(_, _))
      .WillRepeatedly(
          [&](ol_device_iterate_cb_t Callback, void *UserData) -> ol_result_t {
            for (ol_device_handle_t Device : {WithAspect, WithoutAspect})
              std::ignore = Callback(Device, UserData);
            return OL_SUCCESS;
          });
  mockSupport(true, WithAspect);
  mockSupport(false, WithoutAspect);

  platform Platform = platform::get_platforms().at(0);
  std::vector<device> Devices = Platform.get_devices();
  ASSERT_EQ(Devices.size(), 2u);
  for (const device &Device : Devices)
    EXPECT_EQ(Device.has(GetParam().Aspect),
              detail::getSyclObjImpl(Device)->getOLHandle() == WithAspect);
  EXPECT_FALSE(Platform.has(GetParam().Aspect));

  mock::releaseDummyHandles(WithAspect, WithoutAspect, OLPlatform);
}

TEST_P(DeviceHasAspectTest, QueryFailureThrows) {
  EXPECT_CALL(Helper.Mock.get(), olGetDeviceInfo(_, GetParam().Info, _, _))
      .WillOnce(
          Return(Helper.Mock.get().makeEmptyStrError(OL_ERRC_BACKEND_FAILURE)));

  device Device = platform::get_platforms().at(0).get_devices().at(0);
  try {
    std::ignore = Device.has(GetParam().Aspect);
    FAIL() << "device::has() didn't throw";
  } catch (const sycl::exception &E) {
    EXPECT_EQ(E.code(), sycl::errc::runtime);
  }
}

INSTANTIATE_TEST_SUITE_P(
    FloatingPointAspects, DeviceHasAspectTest,
    Values(AspectQuery{aspect::fp16, OL_DEVICE_INFO_HALF_FP_SUPPORT},
           AspectQuery{aspect::fp64, OL_DEVICE_INFO_DOUBLE_FP_SUPPORT}),
    [](const TestParamInfo<AspectQuery> &Info) {
      return Info.param.Aspect == aspect::fp16 ? "fp16" : "fp64";
    });

} // namespace
