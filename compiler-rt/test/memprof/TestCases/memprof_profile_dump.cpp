// RUN: %clangxx_memprof %s -o %t

// RUN: %env_memprof_opts=print_text=true:log_path=stdout %run %t | FileCheck --check-prefix=CHECK-TEXT %s
// RUN: %env_memprof_opts=log_path=stdout %run %t > %t.memprofraw
// RUN: od -c -N 8 %t.memprofraw | FileCheck --check-prefix=CHECK-RAW %s

#include <sanitizer/memprof_interface.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
  char *x = (char *)malloc(10);
  memset(x, 0, 10);
  free(x);
  __memprof_profile_dump();
  x = (char *)malloc(10);
  memset(x, 0, 10);
  free(x);
  return 0;
}
// We should get 2 rounds of profile info, one from the explicit dump request,
// and one at exit.
//
// CHECK-TEXT: Recorded MIBs (incl. live on exit):
// CHECK-TEXT: Memory allocation stack id
// CHECK-TEXT: Stack for id
//
// CHECK-TEXT: Recorded MIBs (incl. live on exit):
// CHECK-TEXT: Memory allocation stack id
// CHECK-TEXT: Stack for id
//
// For the raw profile just check the header magic. The following check assumes that memprof
// runs on little endian architectures.
// CHECK-RAW: 0000000 201   r   f   o   r   p   m 377
