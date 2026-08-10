// RUN: %clangxx_msan -fexceptions -fsanitize-memory-track-origins=2 %libatomic -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-SHADOW

// PPC has no libatomic
// UNSUPPORTED: powerpc64-target-arch
// UNSUPPORTED: powerpc64le-target-arch

#include <sanitizer/msan_interface.h>
#include <stdatomic.h>

typedef struct __attribute((packed)) {
  uint8_t val[3];
} i24;

void copy(i24 *dst, i24 *src);

int main() {
  i24 uninit;
  i24 init = {0};

  __msan_check_mem_is_initialized(&init, 3);
  copy(&init, &uninit);
  __msan_check_mem_is_initialized(&init, 3);
}

void copy(i24 *dst, i24 *src) {
  try {
    __atomic_load(src, dst, __ATOMIC_RELAXED);
  } catch (...) {
  }
}

// CHECK: MemorySanitizer: use-of-uninitialized-value
// CHECK: #0 {{0x[a-f0-9]+}} in main{{.*}}libatomic_load_exceptions.cpp:[[@LINE-11]]

// CHECK-SHADOW: Uninitialized value was stored to memory at
// CHECK-SHADOW: #0 {{0x[a-f0-9]+}} in copy{{.*}}libatomic_load_exceptions.cpp:[[@LINE-9]]
