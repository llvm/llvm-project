// RUN: %clangxx_memprof %s -o %t

// RUN: %env_memprof_opts=print_text=true:log_path=stdout:handle_abort=1 not %run %t 2>&1 | FileCheck --check-prefix=CHECK-TEXT %s

#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
  char *x = (char *)malloc(10);
  memset(x, 0, 10);
  free(x);
  abort();
  return 0;
}

// CHECK-TEXT: MemProfiler:DEADLYSIGNAL
// CHECK-TEXT: Recorded MIBs (incl. live on exit):
// CHECK-TEXT: Memory allocation stack id
// CHECK-TEXT: Stack for id
