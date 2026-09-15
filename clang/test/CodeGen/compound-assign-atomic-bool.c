// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple riscv64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

// When performing compound assignment on atomic_bool, ensure that we
// correctly handle the conversion from integer to boolean, by comparing
// with zero rather than truncating.

// CHECK-LABEL: @compund_assign_add
int compund_assign_add(void) {
    _Atomic _Bool b;

    b += 2;
    // CHECK: add
    // CHECK: icmp ne
    // CHECK-NOT: trunc
    // CHECK: {{cmpxchg|call.*__atomic_compare_exchange}}
    return b;
}

// CHECK-LABEL: @compund_assign_minus
int compund_assign_minus(void) {
    _Atomic _Bool b;

    b -= 2;
    // CHECK: sub
    // CHECK: icmp ne
    // CHECK-NOT: trunc
    // CHECK: {{cmpxchg|call.*__atomic_compare_exchange}}
    return b;
}