// REQUIRES: any-device
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

#include <cassert>

int main() {
  sycl::queue q;

  int *ptr = sycl::malloc_shared<int>(1, q);
  assert(ptr);
  *ptr = 0;

  auto E1 = q.single_task<class DependsOnlyKernel>([=]() { *ptr = 42; });

  // Command group with dependency only and no action.
  auto E2 = q.submit([&](sycl::handler &cgh) { cgh.depends_on(E1); });
  E2.wait();

  assert(*ptr == 42);
  sycl::free(ptr, q);
  return 0;
}
