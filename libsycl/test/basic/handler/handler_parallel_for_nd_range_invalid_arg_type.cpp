// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify \
// RUN: -Xclang -verify-ignore-unexpected=error,note %s

#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;

  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class HandlerNDRangeInvalidArgType>(
        sycl::nd_range<1>{sycl::range<1>{4}, sycl::range<1>{2}},
        [=](sycl::item<1>) {}); // expected-error@* {{must be sycl::nd_item or
                                // be convertible from sycl::nd_item}}
  });

  return 0;
}
