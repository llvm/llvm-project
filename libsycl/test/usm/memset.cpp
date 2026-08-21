// REQUIRES: any-device
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include "include/fill_memset_common.hpp"

int main() {
  queue Q;
  runTests<unsigned char>(Q,
                          [&](void *Ptr) { Q.memset(Ptr, Pattern, DataSize); });
}
