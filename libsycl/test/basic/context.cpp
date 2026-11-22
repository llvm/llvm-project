//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: any-device
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <cstdlib>
#include <iostream>

#include <sycl/sycl.hpp>

using namespace sycl;

void dummyAsyncHandler(sycl::exception_list) {}

void check(const context &Ctx) {
  auto Devices = Ctx.get_devices();

  auto Plt = Ctx.get_platform();
  for (const auto &Dev : Devices) {
    if (Dev.get_platform() != Plt) {
      std::cout << "Device platform does not match context platform"
                << std::endl;
      std::exit(1);
    }
  }
  auto Backend = Ctx.get_backend();
  for (const auto &Dev : Devices) {
    if (Dev.get_backend() != Backend) {
      std::cout << "Device backend does not match context backend" << std::endl;
      std::exit(1);
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

  return 0;
}
