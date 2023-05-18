// RUN: %clangxx_asan %s -o %t
// RUN: %env_asan_opts=print_module_map=1                 not %run %t 2>&1 | FileCheck %s
// RUN: %env_asan_opts=print_module_map=2                 not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan %s -o %t -fsanitize-recover=address
// RUN: %env_asan_opts=print_module_map=2:halt_on_error=0     %run %t 2>&1 | FileCheck %s

// We can't run system("otool") in the simulator.
// UNSUPPORTED: ios

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
  char buf[2048];
  snprintf(buf, sizeof(buf), "otool -l %s 1>&2", argv[0]);
  system(buf);
  // CHECK: cmd LC_UUID
  // CHECK-NEXT: cmdsize 24
  // CHECK-NEXT: uuid [[UUID:[0-9A-F-]{36}]]

  char *x = (char*)malloc(10 * sizeof(char));
  free(x);
  char mybuf[10];
  memcpy(mybuf, x, 10);
  // CHECK: {{.*ERROR: AddressSanitizer: heap-use-after-free on address}}
  // CHECK: Process module map:
  // CHECK: uuid.cpp.tmp {{.*}} <[[UUID]]>

  fprintf(stderr, "Done.\n");
}
