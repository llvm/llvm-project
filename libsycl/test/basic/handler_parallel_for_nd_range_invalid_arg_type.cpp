// RUN: not %clangxx -fsycl -fsyntax-only %s 2>&1 | FileCheck %s

#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;

  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class HandlerNDRangeInvalidArgType>(
        sycl::nd_range<1>{sycl::range<1>{4}, sycl::range<1>{2}},
        [=](sycl::item<1>) {});
  });

  return 0;
}

// CHECK: must be sycl::nd_item or be convertible from sycl::nd_item
