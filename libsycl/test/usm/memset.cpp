// REQUIRES: any-device
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include "Inputs/fill_memset_common.hpp"

#include <climits>

int main() {
  sycl::queue Q;
  runTests<unsigned char>(
      Q, [&](void *Ptr, int Pattern) { Q.memset(Ptr, Pattern, DataSize); });
  // Check that the pattern is truncated to an unsigned char.
  runTests<unsigned char>(
      Q, [&](void *Ptr, int Pattern) { Q.memset(Ptr, Pattern, DataSize); },
      CHAR_MAX + 42);
}
