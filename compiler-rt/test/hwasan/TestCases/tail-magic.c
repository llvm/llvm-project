// Tests free_checks_tail_magic=1.
// RUN: %clang_hwasan %s -o %t
// RUN: %env_hwasan_opts=free_checks_tail_magic=0     %run %t
// RUN: %env_hwasan_opts=free_checks_tail_magic=1 not %run %t 2>&1 | \
// RUN:   FileCheck --check-prefixes=CHECK,CHECK-NONLASTGRANULE --strict-whitespace %s
// RUN:                                           not %run %t 2>&1 | \
// RUN:   FileCheck --check-prefixes=CHECK,CHECK-NONLASTGRANULE --strict-whitespace %s
// RUN: %clang_hwasan -DLAST_GRANULE %s -o %t
// RUN: not %run %t 2>&1 | \
// RUN:   FileCheck --check-prefixes=CHECK,CHECK-LASTGRANULE --strict-whitespace %s

#include <stdlib.h>
#include <stdio.h>
#include <sanitizer/hwasan_interface.h>

static volatile char *sink;

// Overwrite the tail in a non-hwasan function so that we don't detect the
// stores as OOB.
__attribute__((no_sanitize("hwaddress"))) void overwrite_tail() {
#ifdef LAST_GRANULE
  sink[31] = 0x71;
#else // LAST_GRANULE
  sink[20] = 0x42;
  sink[24] = 0x66;
#endif // LAST_GRANULE
}

int main(int argc, char **argv) {
  __hwasan_enable_allocator_tagging();

  char *p = (char*)malloc(20);
  __hwasan_print_shadow(p, 1);
  sink = p;
  overwrite_tail();
  free(p);
// CHECK: HWASan shadow map for {{.*}} (pointer tag [[TAG:[a-f0-9]+]])
// CHECK: ERROR: HWAddressSanitizer: allocation-tail-overwritten; heap object [{{.*}}) of size 20
// CHECK: Stack of invalid access unknown. Issue detected at deallocation time.
// CHECK: deallocated here:
// CHECK: in main {{.*}}tail-magic.c:[[@LINE-5]]
// CHECK: allocated here:
// CHECK: in main {{.*}}tail-magic.c:[[@LINE-11]]
// CHECK-NONLASTGRANULE: Tail contains: .. .. .. .. 42 {{(([a-f0-9]{2} ){3})}}66
// CHECK-LASTGRANULE: Tail contains: .. .. .. .. {{(([a-f0-9]{2} ?)+)}}71{{ *$}}
// CHECK-NEXT: Expected: {{ +}} .. .. .. .. {{([a-f0-9]{2} )+0?}}[[TAG]]{{ *$}}
// CHECK-NONLASTGRANULE-NEXT: {{ +}}^^{{ +}}^^{{ *$}}
// CHECK-LASTGRANULE-NEXT: {{ +}}^^{{ *$}}
  return 0;
}
