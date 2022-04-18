// CHECK_REGULAR_LOAD_STORE: call void @__asan_loadN
// CHECK_REGULAR_LOAD_STORE: call void @__asan_storeN
// RUN: %clangxx_asan -O2 -fsanitize-address-outline-instrumentation %s -o %t
// RUN: %clangxx_asan -O2 -fsanitize-address-outline-instrumentation %s -o - -S \
// RUN:   -emit-llvm | FileCheck %s --check-prefix=CHECK_REGULAR_LOAD_STORE
// RUN: not %run %t A 2>&1 | FileCheck %s --check-prefix=CHECK_0_BYTES
// RUN: not %run %t B 2>&1 | FileCheck %s --check-prefix=CHECK_0_BYTES
// RUN: not %run %t C 2>&1 | FileCheck %s --check-prefix=CHECK_1_BYTES
// RUN: not %run %t D 2>&1 | FileCheck %s --check-prefix=CHECK_1_BYTES

// CHECK_NOABORT_LOAD_STORE: call void @__asan_loadN_noabort
// CHECK_NOABORT_LOAD_STORE: call void @__asan_storeN_noabort
// RUN: %clangxx_asan -O2 -fsanitize-address-outline-instrumentation %s -o %t \
// RUN:   -mllvm -asan-recover=1
// RUN: %clangxx_asan -O2 -fsanitize-address-outline-instrumentation %s -o - -S \
// RUN:   -mllvm -asan-recover=1 -emit-llvm \
// RUN:   | FileCheck %s --check-prefix=CHECK_NOABORT_LOAD_STORE
// RUN: not %run %t A 2>&1 | FileCheck %s --check-prefix=CHECK_0_BYTES
// RUN: not %run %t B 2>&1 | FileCheck %s --check-prefix=CHECK_0_BYTES
// RUN: not %run %t C 2>&1 | FileCheck %s --check-prefix=CHECK_1_BYTES
// RUN: not %run %t D 2>&1 | FileCheck %s --check-prefix=CHECK_1_BYTES

// CHECK_EXP_LOAD_STORE: call void @__asan_exp_loadN
// CHECK_EXP_LOAD_STORE: call void @__asan_exp_storeN
// RUN: %clangxx_asan -O2 -fsanitize-address-outline-instrumentation %s -o %t \
// RUN:   -mllvm -asan-force-experiment=42
// RUN: %clangxx_asan -O2 -fsanitize-address-outline-instrumentation %s -o - -S \
// RUN:   -mllvm -asan-force-experiment=42 -emit-llvm \
// RUN:   | FileCheck %s --check-prefix=CHECK_EXP_LOAD_STORE
// RUN: not %run %t A 2>&1 | FileCheck %s --check-prefix=CHECK_0_BYTES
// RUN: not %run %t B 2>&1 | FileCheck %s --check-prefix=CHECK_0_BYTES
// RUN: not %run %t C 2>&1 | FileCheck %s --check-prefix=CHECK_1_BYTES
// RUN: not %run %t D 2>&1 | FileCheck %s --check-prefix=CHECK_1_BYTES

// CHECK_0_BYTES: ERROR: AddressSanitizer: global-buffer-overflow on address [[ADDR:.*]] at
// CHECK_0_BYTES: [[ADDR]] is located 0 bytes to the right

// CHECK_1_BYTES: ERROR: AddressSanitizer: global-buffer-overflow on address [[ADDR:.*]] at
// CHECK_1_BYTES: [[ADDR]] is located 1 bytes to the right

#include <sanitizer/asan_interface.h>

#include <stdlib.h>
#include <string.h>

static int64_t mem = -1;
static int64_t *volatile G = &mem;

inline uint16_t UNALIGNED_LOAD(const void *p) {
  uint16_t data;
  memcpy(&data, p, sizeof data);
  return data;
}

inline void UNALIGNED_STORE(uint16_t data, void *p) {
  memcpy(p, &data, sizeof data);
}

int main(int argc, char **argv) {
  if (argc != 2)
    return 1;
  int res = 1;
  switch (argv[1][0]) {
  case 'A':
    res = UNALIGNED_LOAD(reinterpret_cast<char *>(G) + 7);
    break;
  case 'B':
    UNALIGNED_STORE(0, reinterpret_cast<char *>(G) + 7);
    break;
  case 'C':
    res = UNALIGNED_LOAD(reinterpret_cast<char *>(G) + 9);
    break;
  case 'D':
    UNALIGNED_STORE(0, reinterpret_cast<char *>(G) + 9);
    break;
  }
  return res;
}
