// RUN: %clangxx -fsycl -fsycl-device-only -std=c++17 -fsyntax-only %s

#include <sycl/sycl.hpp>

struct SingleTaskKernel {
  void operator()() const {}
};

int main() {
  sycl::queue Q;

  Q.single_task<class QueueSingleTaskNamed>(SingleTaskKernel{});

  Q.submit([&](sycl::handler &CGH) {
    CGH.single_task<class HandlerSingleTaskNamed>(SingleTaskKernel{});
  });

  return 0;
}
