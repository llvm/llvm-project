// Tests free_checks_tail_magic=1.
// RUN: %clang_hwasan  %s -o %t
// RUN: %env_hwasan_opts=free_checks_tail_magic=0     %run %t
// RUN: %env_hwasan_opts=free_checks_tail_magic=1 not %run %t 2>&1 | FileCheck %s
// RUN:                                           not %run %t 2>&1 | FileCheck %s

// REQUIRES: stable-runtime

#include <stdlib.h>
#include <stdio.h>
#include <sanitizer/hwasan_interface.h>

static volatile void *sink;

int main(int argc, char **argv) {
  __hwasan_enable_allocator_tagging();

  char *p = (char*)malloc(20);
  sink = p;
  p[20] = 0x42;
  p[24] = 0x66;
  free(p);
// CHECK: ERROR: HWAddressSanitizer: alocation-tail-overwritten; heap object [{{.*}}) of size 20
// CHECK: in main {{.*}}tail-magic.c:[[@LINE-2]]
// CHECK: allocated here:
// CHECK: in main {{.*}}tail-magic.c:[[@LINE-8]]
// CHECK: Tail contains: .. .. .. .. 42 {{.. .. ..}} 66
}
