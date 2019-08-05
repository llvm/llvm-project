// Check that LSan annotations work fine.
// RUN: %clangxx_asan -O0 %s -o %t && %run %t
// RUN: %clangxx_asan -O3 %s -o %t && %run %t

#include <sanitizer/lsan_interface.h>
#include <stdlib.h>

int main() {
  int *x = new int;
  __lsan_ignore_object(x);
  {
    __lsan::ScopedDisabler disabler;
    double *y = new double;
  }
  return 0;
}
