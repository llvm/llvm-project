// RUN: %clangxx_msan -O0 -g -DPOSITIVE %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: MSAN_OPTIONS=verbosity=1 not %run %t 2>&1 | \
// RUN:     FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-VERBOSE

// RUN: %clangxx_msan -O0 -g %s -o %t && %run %t

#include <sanitizer/msan_interface.h>

int main(void) {
  char p[32] = {};
  __msan_poison(p + 10, 2);

  __msan_check_mem_is_initialized(p, 10);
  __msan_check_mem_is_initialized(p + 12, 30);
#ifdef POSITIVE
  __msan_check_mem_is_initialized(p + 5, 20);
  // CHECK: Uninitialized bytes in __msan_check_mem_is_initialized at offset 5 inside [0x{{.*}}, 20)
  // CHECK-VERBOSE: Shadow map of [0x{{.*}}, 0x{{.*}}), 20 bytes:
  // CHECK-VERBOSE: 0x{{.*}}: ..000000 0000ffff 00000000 00000000
  // CHECK-VERBOSE: 0x{{.*}}: 00000000 00...... ........ ........

  // CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
  // CHECK: #0 0x{{.*}}in main{{.*}}msan_check_mem_is_initialized.cpp:[[@LINE-7]]
#endif
  return 0;
}

