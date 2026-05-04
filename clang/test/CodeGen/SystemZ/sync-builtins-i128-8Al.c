// RUN: %clang_cc1 -triple s390x-linux-gnu -O1 -emit-llvm %s -o - 2>&1 | FileCheck %s
//
// Test that an underaligned 16 byte __sync gives a warning.

#include <stdint.h>

__int128 Ptr __attribute__((aligned(8)));

__int128 f1() {
// CHECK: warning: __sync builtin operation must have natural alignment (consider using __atomic)
  return __sync_fetch_and_add(&Ptr, 1);
}

__int128 f2() {
// CHECK: warning: __sync builtin operation must have natural alignment (consider using __atomic)
  return __sync_sub_and_fetch(&Ptr, 1);
}

__int128 f3() {
// CHECK: warning: __sync builtin operation must have natural alignment (consider using __atomic)
  return __sync_val_compare_and_swap(&Ptr, 0, 1);
}

void f4() {
// CHECK: warning: __sync builtin operation must have natural alignment (consider using __atomic)
  __sync_lock_release(&Ptr);
}
