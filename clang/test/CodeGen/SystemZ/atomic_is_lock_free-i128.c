// RUN: %clang_cc1 -triple s390x-linux-gnu -O1 -emit-llvm %s -o - | FileCheck %s
//
// Test __atomic_is_lock_free() and friends.

#include <stdatomic.h>
#include <stdint.h>

__int128 Int128_Al16 __attribute__((aligned(16)));

// CHECK-LABEL: @fun0
// CHECK:       tail call zeroext i1 @__atomic_is_lock_free
_Bool fun0() {
  return __atomic_is_lock_free(16, &Int128_Al16);
}

// CHECK-LABEL: @fun1
// CHECK:       ret i1 false
_Bool fun1() {
  return __atomic_always_lock_free(16, &Int128_Al16);
}

__int128 Int128_Al8 __attribute__((aligned(8)));

// CHECK-LABEL: @fun2
// CHECK:    call zeroext i1 @__atomic_is_lock_free
_Bool fun2() {
  return __atomic_is_lock_free(16, &Int128_Al8);
}

// CHECK-LABEL: @fun3
// CHECK:    ret i1 false
_Bool fun3() {
  return __atomic_always_lock_free(16, &Int128_Al8);
}

// CHECK-LABEL: @fun4
// CHECK:       ret i1 true
_Bool fun4() {
  return __atomic_is_lock_free(16, 0);
}

// CHECK-LABEL: @fun5
// CHECK:       ret i1 true
_Bool fun5() {
  return __atomic_always_lock_free(16, 0);
}

_Atomic __int128 AtomicI128;

// CHECK-LABEL: @fun6
// CHECK:       ret i1 true
_Bool fun6() {
  return __atomic_is_lock_free(16, &AtomicI128);
}

// CHECK-LABEL: @fun7
// CHECK:       ret i1 true
_Bool fun7() {
  return __atomic_always_lock_free(16, &AtomicI128);
}

// CHECK-LABEL: @fun8
// CHECK:       ret i1 true
_Bool fun8() {
  return atomic_is_lock_free(&AtomicI128);
}

// CHECK-LABEL: @fun9
// CHECK:       ret i1 true
_Bool fun9() {
  return __c11_atomic_is_lock_free(16);
}
