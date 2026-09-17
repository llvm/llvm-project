// REQUIRES: any-device
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <cstdlib>
#include <iostream>

#include <sycl/sycl.hpp>

using namespace sycl;

void returnFail() {
  std::cout << "Failed" << std::endl;
  exit(1);
}

void dummyAsyncHandler(sycl::exception_list) {}

void check(const context &Ctx) {
  auto Devices = Ctx.get_devices();

  auto Plt = Ctx.get_platform();
  for (const auto &Dev : Devices) {
    if (Dev.get_platform() != Plt) {
      std::cout << "Device platform does not match context platform"
                << std::endl;
      returnFail();
    }
  }
  auto Backend = Ctx.get_backend();
  for (const auto &Dev : Devices) {
    if (Dev.get_backend() != Backend) {
      std::cout << "Device backend does not match context backend" << std::endl;
      returnFail();
    }
  }
}

int main() {
  context Ctx;
  check(Ctx);

  device Dev;
  context Ctx2(Dev);
  check(Ctx2);

  device Dev2;

  platform Plt = Dev.get_platform();
  context Ctx3(Plt);
  check(Ctx3);

  context Ctx4({Dev, Dev2}, dummyAsyncHandler, {});
  check(Ctx4);

  std::cout << "Passed" << std::endl;
  return 0;
}
