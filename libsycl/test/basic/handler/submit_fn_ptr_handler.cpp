// REQUIRES: any-device
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

#include <cassert>

int *p = nullptr;

void foo(sycl::handler &cgh) {
  auto *copy = p;
  cgh.single_task([=]() { *copy = 42; });
}

int main() {
  sycl::queue q;
  p = sycl::malloc_shared<int>(1, q);
  *p = 0;
  q.submit(foo).wait();
  assert(*p == 42);
  sycl::free(p, q);
  return 0;
}
