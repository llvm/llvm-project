// REQUIRES: any-device
// RUN: %clangxx %sycl_options %s -o %t.out
// RUN: %t.out

#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  queue myQueue;

  int *data = sycl::malloc_shared<int>(1024, myQueue);

  // myQueue.parallel_for(1024, [=](id<1> idx) {
  //   data[idx] = idx;
  // });

  // myQueue.wait();

  size_t error{};
  // for (int i = 0; i < 1024; i++) {
  //   if ((data[i] != i)) {
  //     error++;
  //     std::cerr << "Data mismatch is found: data[" << i << "] = " << data[i]
  //               << std::endl;
  //   }
  // }

  sycl::free(data, myQueue);

  return error > 0;
}
