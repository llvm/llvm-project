// RUN: %clangxx_msan -O2 %s -o %t && \
// RUN:     not %run %t 2>&1 | FileCheck %s

#include <stdlib.h>

int *p;

int main() {
  for (int i = 0; i < 3; i++) {
    int x;
    if (i == 0)
      x = 0;
    p = &x;
  }
  return *p; // BOOM
  // CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
  // CHECK:  #0 0x{{.*}} in main {{.*}}loop-scope.cpp:[[@LINE-2]]
}
