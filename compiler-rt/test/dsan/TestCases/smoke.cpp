// RUN: %clangxx_dsan %s -o %t
// RUN: %run %t

#include <cstdlib>

int main() {
  void *p = std::malloc(16);
  std::free(p);
  return 0;
}
