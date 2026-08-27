// REQUIRES: any-device
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <iostream>

#include <sycl/sycl.hpp>

using namespace sycl;

class Kernel1;

bool check(backend be) {
  switch (be) {
  case backend::opencl:
  case backend::level_zero:
  case backend::cuda:
  case backend::hip:
    return true;
  default:
    return false;
  }
}

void return_fail() {
  std::cout << "Failed" << std::endl;
  exit(1);
}

int main() {
  for (const auto &plt : platform::get_platforms()) {
    if (!check(plt.get_backend())) {
      return_fail();
    }

    auto device = plt.get_devices()[0];
    if (device.get_backend() != plt.get_backend()) {
      return_fail();
    }

    queue q(device);
    if (q.get_backend() != plt.get_backend()) {
      return_fail();
    }

    event e = q.single_task<Kernel1>([]() {});
    if (e.get_backend() != plt.get_backend()) {
      return_fail();
    }
  }
  std::cout << "Passed" << std::endl;
  return 0;
}
