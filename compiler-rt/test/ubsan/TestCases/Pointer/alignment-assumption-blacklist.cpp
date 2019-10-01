// RUN: %clang -fsanitize=alignment -fno-sanitize-recover=alignment                           -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption "

// RUN: rm -f %tmp
// RUN: echo "[alignment]" >> %tmp
// RUN: echo "fun:main" >> %tmp
// RUN: %clang -fsanitize=alignment -fno-sanitize-recover=alignment -fsanitize-blacklist=%tmp -O0 %s -o %t && %run %t 2>&1

#include <stdlib.h>

int main(int argc, char* argv[]) {
  char *ptr = (char *)malloc(2);

  __builtin_assume_aligned(ptr + 1, 0x8000);
  // CHECK: {{.*}}alignment-assumption-blacklist.cpp:[[@LINE-1]]:32: runtime error: assumption of 32768 byte alignment for pointer of type 'char *' failed
  // CHECK: 0x{{.*}}: note: address is {{.*}} aligned, misalignment offset is {{.*}} byte

  free(ptr);

  return 0;
}
