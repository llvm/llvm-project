// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s
// rdar://7536390

typedef unsigned __INT32_TYPE__ uint32_t;

unsigned t(uint32_t *ptr, uint32_t val) {
  // CHECK:      @t
  // CHECK: atomicrmw xchg ptr {{.*}} seq_cst, align 4
  return __sync_lock_test_and_set(ptr, val);
}
