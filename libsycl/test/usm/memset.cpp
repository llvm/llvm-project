// REQUIRES: any-device
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include "Inputs/fill_memset_common.hpp"

int main() {
  sycl::queue Q;
  runTests<unsigned char>(Q,
                          [&](void *Ptr) { Q.memset(Ptr, Pattern, DataSize); });
}
