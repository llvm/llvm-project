// Check that LSan annotations work fine.
// RUN: %clangxx_asan -O0 %s -o %t && %run %t
// RUN: %clangxx_asan -O3 %s -o %t && %run %t

#include <sanitizer/lsan_interface.h>
#include <stdlib.h>

int *x, *y, *z;

int main() {
  x = new int;
  __lsan_ignore_object(x);

  {
    __lsan::ScopedDisabler disabler;
    y = new int;
  }

  z = new int;
  __lsan_ignore_object(z - 1);

  x = y = z = nullptr;
  return 0;
}
