// REQUIRES: any-device
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include "include/fill_memset_common.hpp"

int main() {
  queue Q;
  runTests<int>(Q, [&](void *Ptr) { Q.fill(Ptr, Pattern, DataSize); });
}
