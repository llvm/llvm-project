// RUN: %clang_cc1 -triple s390x-linux-gnu -O1 -emit-llvm %s -o - | FileCheck %s
//
// Test __atomic_is_lock_free() and __atomic_always_lock_free() for __int128
// with 16 byte alignment.

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

// Also test these with a 16 byte size and null-pointer.
// CHECK-LABEL: @fun2
// CHECK:       ret i1 true
_Bool fun2() {
  return __atomic_is_lock_free(16, 0);
}

// CHECK-LABEL: @fun3
// CHECK:       ret i1 true
_Bool fun3() {
  return __atomic_always_lock_free(16, 0);
}

// Also test __c11_atomic_is_lock_free() with a 16 byte size.
// CHECK-LABEL: @fun4
// CHECK:       ret i1 true
_Bool fun4() {
  return __c11_atomic_is_lock_free(16);
}

