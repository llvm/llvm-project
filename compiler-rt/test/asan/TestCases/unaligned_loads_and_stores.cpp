// RUN: %clangxx_asan -O0 %s -o %t
// RUN: not %run %t A 2>&1 | FileCheck --check-prefix=CHECK-A %s
// RUN: not %run %t B 2>&1 | FileCheck --check-prefix=CHECK-B %s
// RUN: not %run %t C 2>&1 | FileCheck --check-prefix=CHECK-C %s
// RUN: not %run %t D 2>&1 | FileCheck --check-prefix=CHECK-D %s
// RUN: not %run %t E 2>&1 | FileCheck --check-prefix=CHECK-E %s

// RUN: not %run %t K 2>&1 | FileCheck --check-prefix=CHECK-K %s
// RUN: not %run %t L 2>&1 | FileCheck --check-prefix=CHECK-L %s
// RUN: not %run %t M 2>&1 | FileCheck --check-prefix=CHECK-M %s
// RUN: not %run %t N 2>&1 | FileCheck --check-prefix=CHECK-N %s
// RUN: not %run %t O 2>&1 | FileCheck --check-prefix=CHECK-O %s

#include <sanitizer/asan_interface.h>

#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
  if (argc != 2) return 1;
  char *x = new char[16];
  memset(x, 0xab, 16);
  int res = 1;
  switch (argv[1][0]) {
    case 'A': res = __sanitizer_unaligned_load16(x + 15); break;
//  CHECK-A ERROR: AddressSanitizer: heap-buffer-overflow on address
//  CHECK-A: main{{.*}}unaligned_loads_and_stores.cpp:[[@LINE-2]]
//  CHECK-A: is located 0 bytes to the right of 16-byte region
    case 'B': res = __sanitizer_unaligned_load32(x + 14); break;
//  CHECK-B: main{{.*}}unaligned_loads_and_stores.cpp:[[@LINE-1]]
    case 'C': res = __sanitizer_unaligned_load32(x + 13); break;
//  CHECK-C: main{{.*}}unaligned_loads_and_stores.cpp:[[@LINE-1]]
    case 'D': res = __sanitizer_unaligned_load64(x + 15); break;
//  CHECK-D: main{{.*}}unaligned_loads_and_stores.cpp:[[@LINE-1]]
    case 'E': res = __sanitizer_unaligned_load64(x + 9); break;
//  CHECK-E: main{{.*}}unaligned_loads_and_stores.cpp:[[@LINE-1]]

    case 'K': __sanitizer_unaligned_store16(x + 15, 0); break;
//  CHECK-K ERROR: AddressSanitizer: heap-buffer-overflow on address
//  CHECK-K: main{{.*}}unaligned_loads_and_stores.cpp:[[@LINE-2]]
//  CHECK-K: is located 0 bytes to the right of 16-byte region
    case 'L': __sanitizer_unaligned_store32(x + 15, 0); break;
//  CHECK-L: main{{.*}}unaligned_loads_and_stores.cpp:[[@LINE-1]]
    case 'M': __sanitizer_unaligned_store32(x + 13, 0); break;
//  CHECK-M: main{{.*}}unaligned_loads_and_stores.cpp:[[@LINE-1]]
    case 'N': __sanitizer_unaligned_store64(x + 10, 0); break;
//  CHECK-N: main{{.*}}unaligned_loads_and_stores.cpp:[[@LINE-1]]
    case 'O': __sanitizer_unaligned_store64(x + 14, 0); break;
//  CHECK-O: main{{.*}}unaligned_loads_and_stores.cpp:[[@LINE-1]]
  }
  delete x;
  return res;
}
