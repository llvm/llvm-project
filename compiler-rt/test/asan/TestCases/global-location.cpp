// RUN: %clangxx_asan -g -O2 %s -o %t
// RUN: not %run %t g 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=GLOB
// RUN: not %run %t c 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=CLASS_STATIC
// RUN: not %run %t f 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=FUNC_STATIC
// RUN: not %run %t l 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=LITERAL

// COFF doesn't support debuginfo for globals. For the non-debuginfo tests, see global-location-nodebug.cpp.
// XFAIL: windows-msvc

// CHECK: AddressSanitizer: global-buffer-overflow

#include <string.h>

struct C {
  static int array[10];
  // CLASS_STATIC:      0x{{.*}} is located 4 bytes to the right of global variable 'C::array' defined in '{{.*}}global-location.cpp:[[@LINE-1]]' {{.*}} of size 40
};

int global[10];
// GLOB:      0x{{.*}} is located 4 bytes to the right of global variable 'global' defined in '{{.*}}global-location.cpp:[[@LINE-1]]' {{.*}} of size 40
int C::array[10];

int main(int argc, char **argv) {
  int one = argc - 1;
  switch (argv[1][0]) {
  case 'g': return global[one * 11];
  case 'c': return C::array[one * 11];
  case 'f':
    static int array[10];
    // FUNC_STATIC:      0x{{.*}} is located 4 bytes to the right of global variable 'array' defined in '{{.*}}global-location.cpp:[[@LINE-1]]' {{.*}} of size 40
    memset(array, 0, 10);
    return array[one * 11];
  case 'l':
    const char *str = "0123456789";
    // LITERAL:      0x{{.*}} is located 0 bytes to the right of global variable {{.*}} defined in '{{.*}}global-location.cpp:[[@LINE-1]]' {{.*}} of size 11
    return str[one * 11];
  }
  return 0;
}

// CHECK: SUMMARY: AddressSanitizer: global-buffer-overflow
