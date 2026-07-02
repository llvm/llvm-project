// REQUIRES: any-device
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

#include <cassert>

int main() {
  sycl::queue q;

  constexpr size_t N = 16;
  int *data = sycl::malloc_shared<int>(N, q);
  int *token = sycl::malloc_shared<int>(1, q);
  assert(data && token);

  for (size_t i = 0; i < N; ++i)
    data[i] = 0;
  *token = 0;

  auto initEvent =
      q.single_task<class HandlerNDRangeDependsInit>([=]() { *token = 9; });

  auto kernelEvent = q.submit([&](sycl::handler &cgh) {
    cgh.depends_on(initEvent);
    cgh.parallel_for<class HandlerNDRangeDependsKernel>(
        sycl::nd_range<1>{sycl::range<1>{N}, sycl::range<1>{4}},
        [=](sycl::nd_item<1> it) {
          const size_t i = it.get_global_id(0);
          data[i] = static_cast<int>(i) + *token;
        });
  });

  kernelEvent.wait();

  for (size_t i = 0; i < N; ++i)
    assert(data[i] == static_cast<int>(i) + 9);

  sycl::free(data, q);
  sycl::free(token, q);
  return 0;
}
