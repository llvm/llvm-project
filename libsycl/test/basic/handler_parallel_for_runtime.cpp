// REQUIRES: any-device
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

#include <cassert>

int main() {
  sycl::queue q;

  constexpr size_t N = 16;
  int *data = sycl::malloc_shared<int>(N, q);
  assert(data);

  for (size_t i = 0; i < N; ++i)
    data[i] = 0;

  q.submit([&](sycl::handler &cgh) {
     cgh.parallel_for<class HandlerParallelForRuntime>(
         sycl::range<1>{N},
         [=](sycl::item<1> it) { data[it[0]] = static_cast<int>(it[0]) + 7; });
   }).wait();

  for (size_t i = 0; i < N; ++i)
    assert(data[i] == static_cast<int>(i) + 7);

  sycl::free(data, q);
  return 0;
}
