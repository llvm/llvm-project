// REQUIRES: any-device
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

// Adapted from upstream SYCL tests that validate empty command-group
// dependency behavior and handler memory operations.

#include <sycl/sycl.hpp>

#include <cassert>

int main() {
  sycl::queue q;

  int *src = sycl::malloc_shared<int>(4, q);
  int *dst = sycl::malloc_shared<int>(4, q);
  assert(src && dst);
  src[0] = 1;
  src[1] = 2;
  src[2] = 3;
  src[3] = 4;
  dst[0] = 0;
  dst[1] = 0;
  dst[2] = 0;
  dst[3] = 0;

  // Exercise handler::memcpy in a submitted command group.
  auto memcpy_event = q.submit(
      [&](sycl::handler &cgh) { cgh.memcpy(dst, src, 4 * sizeof(int)); });

  auto *shared_value = sycl::malloc_shared<int>(1, q);
  assert(shared_value);
  *shared_value = 0;

  auto kernel_event = q.submit([&](sycl::handler &cgh) {
    cgh.depends_on(memcpy_event);
    cgh.single_task<class DependsOnKernel>([=]() { *shared_value = dst[0]; });
  });

  kernel_event.wait();

  assert(dst[0] == 1 && dst[1] == 2 && dst[2] == 3 && dst[3] == 4);
  assert(*shared_value == 1);

  sycl::free(src, q);
  sycl::free(dst, q);
  sycl::free(shared_value, q);
  return 0;
}