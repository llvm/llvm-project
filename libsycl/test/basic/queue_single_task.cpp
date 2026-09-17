// REQUIRES: any-device
// RUN: %clangxx -fsycl  %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

class Test;

int main() {
  sycl::queue Q;
  int *Ptr = sycl::malloc_shared<int>(1, Q);
  *Ptr = 0;
  Q.single_task<Test>([=]() { *Ptr = 42; });
  Q.wait();

  bool Failed = *Ptr != 42;

  sycl::free(Ptr, Q);
  return Failed;
}
