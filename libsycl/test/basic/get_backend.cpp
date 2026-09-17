// REQUIRES: any-device
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <cstdlib>
#include <iostream>

#include <sycl/sycl.hpp>

using namespace sycl;

class Kernel1;

bool check(backend Backend) {
  switch (Backend) {
  case backend::opencl:
  case backend::level_zero:
  case backend::cuda:
  case backend::hip:
    return true;
  default:
    return false;
  }
}

void returnFail() {
  std::cout << "Failed" << std::endl;
  exit(1);
}

int main() {
  for (const auto &Plt : platform::get_platforms()) {
    if (!check(Plt.get_backend())) {
      returnFail();
    }

    auto Dev = Plt.get_devices()[0];
    if (Dev.get_backend() != Plt.get_backend()) {
      returnFail();
    }

    queue Q(Dev);
    if (Q.get_backend() != Plt.get_backend()) {
      returnFail();
    }

    event E = Q.single_task<Kernel1>([]() {});
    if (E.get_backend() != Plt.get_backend()) {
      returnFail();
    }
  }
  std::cout << "Passed" << std::endl;
  return 0;
}
