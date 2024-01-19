// RUN: %clang_cc1 -triple s390x-linux-gnu -O1 -emit-llvm %s -o - | FileCheck %s
//
// Test __atomic_is_lock_free() and friends.

#include <stdatomic.h>
#include <stdint.h>

typedef __attribute__((aligned(16))) __int128 __int128_Al16;

_Atomic __int128 Int128_Atomic;
__int128_Al16    Int128_Al16;
__int128         Int128;
struct { int I[3]; } _Atomic AtomicStruct;
_Atomic long double Atomic_fp128; // Also check the alignment of this.

// Check alignments of the variables. @AtomicStruct gets padded and its size
// and alignment becomes 16. Only a power-of-2 size is considered, so 16 (not
// 12) needs to be specified with the intrinsics below.
//
// CHECK: %struct.anon = type { [3 x i32] }
// CHECK: @Int128 = {{.*}} i128 0, align 8
// CHECK: @Int128_Atomic = {{.*}} i128 0, align 16
// CHECK: @Int128_Al16 = {{.*}} i128 0, align 16
// CHECK: @AtomicStruct = {{.*}} { %struct.anon, [4 x i8] } zeroinitializer, align 16
// CHECK: @Atomic_fp128 = {{.*}} fp128 0xL00000000000000000000000000000000, align 16


// CHECK-LABEL: @fun0
// CHECK:       ret i1 true
_Bool fun0() {
  return __atomic_is_lock_free(16, &Int128_Atomic);
}

// CHECK-LABEL: @fun1
// CHECK:       ret i1 true
_Bool fun1() {
  return __atomic_always_lock_free(16, &Int128_Atomic);
}

// CHECK-LABEL: @fun2
// CHECK:       ret i1 true
_Bool fun2() {
  return __atomic_is_lock_free(16, &Int128_Al16);
}

// CHECK-LABEL: @fun3
// CHECK:       ret i1 true
_Bool fun3() {
  return __atomic_always_lock_free(16, &Int128_Al16);
}

// CHECK-LABEL: @fun4
// CHECK:    call zeroext i1 @__atomic_is_lock_free
_Bool fun4() {
  return __atomic_is_lock_free(16, &Int128);
}

// CHECK-LABEL: @fun5
// CHECK:    ret i1 false
_Bool fun5() {
  return __atomic_always_lock_free(16, &Int128);
}

// CHECK-LABEL: @fun6
// CHECK:       ret i1 true
_Bool fun6() {
  return __atomic_is_lock_free(16, 0);
}

// CHECK-LABEL: @fun7
// CHECK:       ret i1 true
_Bool fun7() {
  return __atomic_always_lock_free(16, 0);
}

// CHECK-LABEL: @fun8
// CHECK:       ret i1 true
_Bool fun8() {
  return __atomic_is_lock_free(16, &AtomicStruct);
}

// CHECK-LABEL: @fun9
// CHECK:       ret i1 true
_Bool fun9() {
  return __atomic_always_lock_free(16, &AtomicStruct);
}

// CHECK-LABEL: @fun10
// CHECK:       ret i1 true
_Bool fun10() {
  return atomic_is_lock_free(&Int128_Atomic);
}

// CHECK-LABEL: @fun11
// CHECK:       ret i1 true
_Bool fun11() {
  return __c11_atomic_is_lock_free(16);
}
