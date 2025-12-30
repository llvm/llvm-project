// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

// When performing compound assignment on atomic_bool, ensure that we
// correctly handle the conversion from integer to boolean, by comparing
// with zero rather than truncating.

#include <stdatomic.h>
#include <stdbool.h>


// CHECK: @compund_assign_add
int compund_assign_add(void) {
    atomic_bool b;

    b += 2;
    // CHECK: add
    // CHECK-NEXT: icmp ne
    // CHECK-NEXT: zext
    // CHECK-NEXT: cmpxchg
    return b;
}

// CHECK: @compund_assign_minus
int compund_assign_minus(void) {
    atomic_bool b;

    b -= 2;
    // CHECK: sub
    // CHECK-NEXT: icmp ne
    // CHECK-NEXT: zext
    // CHECK-NEXT: cmpxchg
    return b;
}
