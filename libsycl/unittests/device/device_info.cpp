//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Checks that device information descriptors backed by liboffload are queried
// through the matching OL_DEVICE_INFO_* enumerator and that the returned
// values are converted to the SYCL return types.

#include <common/unittests_helper.hpp>

#include <sycl/sycl.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace sycl;
using namespace ::testing;

namespace {

template <typename T>
void mockDeviceInfo(mock::MockLiboffload &Mock, ol_device_info_t Prop,
                    T Value) {
  // Queries with fixed length results should not request size.
  EXPECT_CALL(Mock, olGetDeviceInfoSize(_, Prop, _)).Times(0);
  ON_CALL(Mock, olGetDeviceInfo(_, Prop, _, _))
      .WillByDefault([Value](ol_device_handle_t, ol_device_info_t,
                             size_t PropSize, void *PropValue) -> ol_result_t {
        EXPECT_EQ(PropSize, sizeof(T));
        std::memcpy(PropValue, &Value, sizeof(T));
        return OL_SUCCESS;
      });
}

void mockDeviceInfoString(mock::MockLiboffload &Mock, ol_device_info_t Prop,
                          std::string Value) {
  // liboffload counts the null terminator in the size.
  const size_t Size = Value.size() + 1;
  EXPECT_CALL(Mock, olGetDeviceInfoSize(_, Prop, _))
      .WillOnce([Size](ol_device_handle_t, ol_device_info_t,
                       size_t *PropSizeRet) -> ol_result_t {
        *PropSizeRet = Size;
        return OL_SUCCESS;
      });
  EXPECT_CALL(Mock, olGetDeviceInfo(_, Prop, _, _))
      .WillOnce([Value, Size](ol_device_handle_t, ol_device_info_t,
                              size_t PropSize, void *PropValue) -> ol_result_t {
        EXPECT_EQ(PropSize, Size);
        std::memcpy(PropValue, Value.c_str(), Size);
        return OL_SUCCESS;
      });
}

class DeviceInfoTest : public Test {
protected:
  void SetUp() override {
    auto Platforms = platform::get_platforms();
    ASSERT_EQ(Platforms.size(), 1u);
    auto Devices = Platforms[0].get_devices();
    ASSERT_EQ(Devices.size(), 1u);
    Device = Devices[0];
  }

  mock::MockLiboffload &getMock() { return Helper.Mock.get(); }

  unittests::UnittestsHelper Helper;
  device Device;
};

} // namespace

TEST_F(DeviceInfoTest, VendorId) {
  constexpr uint32_t VendorId = 0x8086;
  mockDeviceInfo<uint32_t>(getMock(), OL_DEVICE_INFO_VENDOR_ID, VendorId);
  EXPECT_EQ(Device.get_info<info::device::vendor_id>(), VendorId);
}

TEST_F(DeviceInfoTest, MaxComputeUnits) {
  constexpr uint32_t ComputeUnits = 96;
  mockDeviceInfo<uint32_t>(getMock(), OL_DEVICE_INFO_NUM_COMPUTE_UNITS,
                           ComputeUnits);
  EXPECT_EQ(Device.get_info<info::device::max_compute_units>(), ComputeUnits);
}

TEST_F(DeviceInfoTest, MaxWorkItemDimensions) {
  EXPECT_EQ(Device.get_info<info::device::max_work_item_dimensions>(), 3u);
}

TEST_F(DeviceInfoTest, MaxWorkItemSizes) {
  constexpr uint32_t X = 1024, Y = 512, Z = 64;
  mockDeviceInfo<ol_dimensions_t>(
      getMock(), OL_DEVICE_INFO_MAX_WORK_GROUP_SIZE_PER_DIMENSION,
      ol_dimensions_t{X, Y, Z});

  // SYCL linearizes ranges with the last dimension varying fastest, so the
  // liboffload dimensions should be reversed.
  EXPECT_EQ(Device.get_info<info::device::max_work_item_sizes<1>>(),
            range<1>(X));
  EXPECT_EQ(Device.get_info<info::device::max_work_item_sizes<2>>(),
            range<2>(Y, X));
  EXPECT_EQ(Device.get_info<info::device::max_work_item_sizes<3>>(),
            range<3>(Z, Y, X));
}

TEST_F(DeviceInfoTest, MaxWorkGroupSize) {
  // liboffload reports uint32_t while SYCL returns size_t.
  constexpr uint32_t WGSize = 1024;
  mockDeviceInfo<uint32_t>(getMock(), OL_DEVICE_INFO_MAX_WORK_GROUP_SIZE,
                           WGSize);
  auto Result = Device.get_info<info::device::max_work_group_size>();
  static_assert(std::is_same_v<decltype(Result), std::size_t>);
  EXPECT_EQ(Result, WGSize);
}

TEST_F(DeviceInfoTest, NativeVectorWidths) {
  constexpr uint32_t CharWidth = 16;
  constexpr uint32_t ShortWidth = 8;
  constexpr uint32_t IntWidth = 4;
  constexpr uint32_t LongWidth = 2;
  constexpr uint32_t FloatWidth = 5;
  constexpr uint32_t DoubleWidth = 1;
  constexpr uint32_t HalfWidth = 32;

  mockDeviceInfo<uint32_t>(getMock(), OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR,
                           CharWidth);
  mockDeviceInfo<uint32_t>(getMock(), OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT,
                           ShortWidth);
  mockDeviceInfo<uint32_t>(getMock(), OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT,
                           IntWidth);
  mockDeviceInfo<uint32_t>(getMock(), OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG,
                           LongWidth);
  mockDeviceInfo<uint32_t>(getMock(), OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT,
                           FloatWidth);
  mockDeviceInfo<uint32_t>(getMock(), OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE,
                           DoubleWidth);
  mockDeviceInfo<uint32_t>(getMock(), OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF,
                           HalfWidth);

  EXPECT_EQ(Device.get_info<info::device::native_vector_width_char>(),
            CharWidth);
  EXPECT_EQ(Device.get_info<info::device::native_vector_width_short>(),
            ShortWidth);
  EXPECT_EQ(Device.get_info<info::device::native_vector_width_int>(), IntWidth);
  EXPECT_EQ(Device.get_info<info::device::native_vector_width_long>(),
            LongWidth);
  EXPECT_EQ(Device.get_info<info::device::native_vector_width_float>(),
            FloatWidth);
  EXPECT_EQ(Device.get_info<info::device::native_vector_width_double>(),
            DoubleWidth);
  EXPECT_EQ(Device.get_info<info::device::native_vector_width_half>(),
            HalfWidth);

  EXPECT_EQ(Device.get_info<info::device::preferred_vector_width_char>(),
            CharWidth);
  EXPECT_EQ(Device.get_info<info::device::preferred_vector_width_short>(),
            ShortWidth);
  EXPECT_EQ(Device.get_info<info::device::preferred_vector_width_int>(),
            IntWidth);
  EXPECT_EQ(Device.get_info<info::device::preferred_vector_width_long>(),
            LongWidth);
  EXPECT_EQ(Device.get_info<info::device::preferred_vector_width_float>(),
            FloatWidth);
  EXPECT_EQ(Device.get_info<info::device::preferred_vector_width_double>(),
            DoubleWidth);
  EXPECT_EQ(Device.get_info<info::device::preferred_vector_width_half>(),
            HalfWidth);
}

TEST_F(DeviceInfoTest, MaxClockFrequency) {
  constexpr uint32_t ClockFrequency = 2100;
  mockDeviceInfo<uint32_t>(getMock(), OL_DEVICE_INFO_MAX_CLOCK_FREQUENCY,
                           ClockFrequency);
  EXPECT_EQ(Device.get_info<info::device::max_clock_frequency>(),
            ClockFrequency);
}

TEST_F(DeviceInfoTest, AddressBits) {
  constexpr uint32_t AddressBits = 64;
  mockDeviceInfo<uint32_t>(getMock(), OL_DEVICE_INFO_ADDRESS_BITS, AddressBits);
  EXPECT_EQ(Device.get_info<info::device::address_bits>(), AddressBits);
}

TEST_F(DeviceInfoTest, MemorySizes) {
  // Values above 4 GiB make sure 64-bit results are not truncated.
  constexpr uint64_t MaxAlloc = 0x1'0000'0000ull + 1;
  constexpr uint64_t GlobalMem = 0x4'0000'0000ull;
  constexpr uint64_t LocalMem = 64 * 1024;
  mockDeviceInfo<uint64_t>(getMock(), OL_DEVICE_INFO_MAX_MEM_ALLOC_SIZE,
                           MaxAlloc);
  mockDeviceInfo<uint64_t>(getMock(), OL_DEVICE_INFO_GLOBAL_MEM_SIZE,
                           GlobalMem);
  mockDeviceInfo<uint64_t>(getMock(), OL_DEVICE_INFO_WORK_GROUP_LOCAL_MEM_SIZE,
                           LocalMem);

  EXPECT_EQ(Device.get_info<info::device::max_mem_alloc_size>(), MaxAlloc);
  EXPECT_EQ(Device.get_info<info::device::global_mem_size>(), GlobalMem);
  EXPECT_EQ(Device.get_info<info::device::local_mem_size>(), LocalMem);
}

constexpr ol_device_fp_capability_flags_t AllOLFPFlags =
    OL_DEVICE_FP_CAPABILITY_FLAG_CORRECTLY_ROUNDED_DIVIDE_SQRT |
    OL_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_NEAREST |
    OL_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_ZERO |
    OL_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_INF |
    OL_DEVICE_FP_CAPABILITY_FLAG_INF_NAN | OL_DEVICE_FP_CAPABILITY_FLAG_DENORM |
    OL_DEVICE_FP_CAPABILITY_FLAG_FMA | OL_DEVICE_FP_CAPABILITY_FLAG_SOFT_FLOAT;

TEST_F(DeviceInfoTest, FPConfig) {
  mockDeviceInfo<ol_device_fp_capability_flags_t>(
      getMock(), OL_DEVICE_INFO_HALF_FP_CONFIG,
      OL_DEVICE_FP_CAPABILITY_FLAG_INF_NAN |
          OL_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_NEAREST);
  mockDeviceInfo<ol_device_fp_capability_flags_t>(
      getMock(), OL_DEVICE_INFO_SINGLE_FP_CONFIG, AllOLFPFlags);
  mockDeviceInfo<ol_device_fp_capability_flags_t>(
      getMock(), OL_DEVICE_INFO_DOUBLE_FP_CONFIG, 0);

  EXPECT_THAT(Device.get_info<info::device::half_fp_config>(),
              UnorderedElementsAre(info::fp_config::inf_nan,
                                   info::fp_config::round_to_nearest));
  EXPECT_THAT(Device.get_info<info::device::single_fp_config>(),
              UnorderedElementsAre(
                  info::fp_config::correctly_rounded_divide_sqrt,
                  info::fp_config::round_to_nearest,
                  info::fp_config::round_to_zero, info::fp_config::round_to_inf,
                  info::fp_config::inf_nan, info::fp_config::denorm,
                  info::fp_config::fma, info::fp_config::soft_float));
  EXPECT_THAT(Device.get_info<info::device::double_fp_config>(), IsEmpty());
}

TEST_F(DeviceInfoTest, DoubleFPConfigWithoutCorrectlyRoundedDivideSqrt) {
  // correctly_rounded_divide_sqrt is not defined for double precision, so it
  // is dropped from double_fp_config even if liboffload reports it.
  mockDeviceInfo<ol_device_fp_capability_flags_t>(
      getMock(), OL_DEVICE_INFO_DOUBLE_FP_CONFIG, AllOLFPFlags);

  EXPECT_THAT(Device.get_info<info::device::double_fp_config>(),
              UnorderedElementsAre(
                  info::fp_config::round_to_nearest,
                  info::fp_config::round_to_zero, info::fp_config::round_to_inf,
                  info::fp_config::inf_nan, info::fp_config::denorm,
                  info::fp_config::fma, info::fp_config::soft_float));
}

TEST_F(DeviceInfoTest, Strings) {
  constexpr const char *Name = "Mock GPU";
  constexpr const char *Vendor = "Mock Vendor";
  constexpr const char *DriverVersion = "1.2.3";
  mockDeviceInfoString(getMock(), OL_DEVICE_INFO_PRODUCT_NAME, Name);
  mockDeviceInfoString(getMock(), OL_DEVICE_INFO_VENDOR, Vendor);
  mockDeviceInfoString(getMock(), OL_DEVICE_INFO_DRIVER_VERSION, DriverVersion);

  EXPECT_EQ(Device.get_info<info::device::name>(), Name);
  EXPECT_EQ(Device.get_info<info::device::vendor>(), Vendor);
  EXPECT_EQ(Device.get_info<info::device::driver_version>(), DriverVersion);
}

TEST_F(DeviceInfoTest, Platform) {
  EXPECT_EQ(Device.get_info<info::device::platform>(), Device.get_platform());
}
