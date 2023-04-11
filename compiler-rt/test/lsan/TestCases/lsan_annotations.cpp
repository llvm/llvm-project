// Check that LSan annotations work fine.
// RUN: %clangxx_lsan -O0 %s -o %t && %run %t
// RUN: %clangxx_lsan -O3 %s -o %t && %run %t

#include <sanitizer/lsan_interface.h>
#include <stdlib.h>

int *x, *y, *z;

int main() {
  x = new int;
  __lsan_ignore_object(x);

  z = new int[1000000];  // Large enough for the secondary allocator.
  __lsan_ignore_object(z);

  {
    __lsan::ScopedDisabler disabler;
    y = new int;
  }

  x = y = nullptr;
  return 0;
}
