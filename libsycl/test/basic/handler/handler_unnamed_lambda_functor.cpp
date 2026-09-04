// RUN: %clangxx -fsycl -fsycl-device-only -std=c++17 -fsyntax-only %s

#include <sycl/sycl.hpp>

struct SingleTaskKernel {
  void operator()() const {}
};

int main() {
  sycl::queue q;

  q.single_task<class QueueSingleTaskNamed>(SingleTaskKernel{});

  q.submit([&](sycl::handler &cgh) {
    cgh.single_task<class HandlerSingleTaskNamed>(SingleTaskKernel{});
  });

  return 0;
}
