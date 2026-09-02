#include <mock/helpers.hpp>

#include <sycl/__impl/context.hpp>
#include <sycl/__impl/device.hpp>
#include <sycl/__impl/exception.hpp>
#include <sycl/__impl/platform.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <utility>

using namespace sycl;
using namespace ::testing;

void dummyAsyncHandler(exception_list) {}

TEST(Context, DefaultConstructor) {
  mock::MockWrapper Mock;

  // TODO: remove once context is properly implemented
  std::ignore = device{};

  EXPECT_CALL(Mock.get(), olCreateContext(_, _, _)).Times(1);
  EXPECT_CALL(Mock.get(), olDestroyContext(_)).Times(1);

  context Ctx;
}

TEST(Context, DeviceConstructor) {
  mock::MockWrapper Mock;
  device Dev;

  EXPECT_CALL(Mock.get(), olCreateContext(_, _, _)).Times(1);
  EXPECT_CALL(Mock.get(), olDestroyContext(_)).Times(1);

  context Ctx(Dev);
}

TEST(Context, DeviceConstructorWithAsyncHandler) {
  mock::MockWrapper Mock;
  device Dev;
  async_handler AsyncHandler = dummyAsyncHandler;

  EXPECT_CALL(Mock.get(), olCreateContext(_, _, _)).Times(1);
  EXPECT_CALL(Mock.get(), olDestroyContext(_)).Times(1);

  context Ctx(Dev, AsyncHandler);
}

TEST(Context, PlatformConstructor) {
  mock::MockWrapper Mock;
  device Dev;
  platform Plt = Dev.get_platform();

  EXPECT_CALL(Mock.get(), olCreateContext(_, _, _)).Times(1);
  EXPECT_CALL(Mock.get(), olDestroyContext(_)).Times(1);

  context Ctx(Plt);
}

TEST(Context, DeviceListConstructor) {
  mock::MockWrapper Mock;
  device Dev1;
  device Dev2;
  async_handler AsyncHandler = dummyAsyncHandler;

  EXPECT_CALL(Mock.get(), olCreateContext(_, _, _)).Times(1);
  EXPECT_CALL(Mock.get(), olDestroyContext(_)).Times(1);

  context Ctx({Dev1, Dev2}, AsyncHandler);
}

TEST(Context, DeviceListConstructorThrowsOnEmptyList) {
  mock::MockWrapper Mock;
  async_handler AsyncHandler = dummyAsyncHandler;

  EXPECT_CALL(Mock.get(), olCreateContext(_, _, _)).Times(0);

  try {
    context Ctx(std::vector<device>{}, AsyncHandler);
    FAIL() << "Expected sycl::exception";
  } catch (const exception &Ex) {
    EXPECT_EQ(Ex.code(), make_error_code(errc::invalid));
  }
}
