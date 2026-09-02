// REQUIRES: any-device
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <iostream>

#include <sycl/sycl.hpp>

using namespace sycl;

void return_fail() {
  std::cout << "Failed" << std::endl;
  exit(1);
}

void dummyAsyncHandler(sycl::exception_list) {}

void check(const context &ctx) {
  auto devices = ctx.get_devices();

  auto plt = ctx.get_platform();
  for (const auto &dev : devices) {
    if (dev.get_platform() != plt) {
      std::cout << "Device platform does not match context platform"
                << std::endl;
      return_fail();
    }
  }
  auto backend = ctx.get_backend();
  for (const auto &dev : devices) {
    if (dev.get_backend() != backend) {
      std::cout << "Device backend does not match context backend" << std::endl;
      return_fail();
    }
  }
}

int main() {
  context ctx;
  check(ctx);

  device dev;
  context ctx2(dev);
  check(ctx2);

  device dev2;

  platform plt = dev.get_platform();
  context ctx3(plt);
  check(ctx3);

  context ctx4({dev, dev2}, dummyAsyncHandler,
               {/* explicit properties list */});
  check(ctx4);

  std::cout << "Passed" << std::endl;
  return 0;
}
