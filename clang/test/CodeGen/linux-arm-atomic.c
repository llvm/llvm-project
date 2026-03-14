// RUN: %clang_cc1 %s -emit-llvm -o - -triple=armv7-unknown-linux | FileCheck %s
// RUN: %clang_cc1 %s -emit-llvm -o - -triple=armv6-unknown-linux | FileCheck %s
// RUN: %clang_cc1 %s -emit-llvm -o - -triple=thumbv7-unknown-linux | FileCheck %s
// RUN: %clang_cc1 %s -emit-llvm -o - -triple=armv6-unknown-freebsd | FileCheck %s

typedef int _Atomic_word;
_Atomic_word exchange_and_add(volatile _Atomic_word *__mem, int __val) {
  return __atomic_fetch_add(__mem, __val, __ATOMIC_ACQ_REL);
}

// CHECK: define {{.*}} @exchange_and_add
// CHECK: atomicrmw {{.*}} add ptr {{.*}} acq_rel, align 4
