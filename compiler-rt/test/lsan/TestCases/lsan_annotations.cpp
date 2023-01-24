// Check that LSan annotations work fine.
// RUN: %clangxx_lsan -O0 %s -o %t && %run %t
// RUN: %clangxx_lsan -O3 %s -o %t && %run %t

#include <sanitizer/lsan_interface.h>
#include <stdlib.h>

int *x, *y;

int main() {
  x = new int;
  __lsan_ignore_object(x);

  {
    __lsan::ScopedDisabler disabler;
    y = new int;
  }

  x = y = nullptr;
  return 0;
}
