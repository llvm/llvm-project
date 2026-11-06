// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s
// RUN: %if cir-enabled %{ %clang_cc1 -fclangir -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,CIR %}

long fn(void) { return __builtin_bit_cast(long, (long)&fn); }

// CHECK-LABEL: define{{.*}} i64 @fn()
// capture the CIR 'ret'.
// CIR:   alloca i64
// CHECK: [[TMP:%.*]] = alloca i64
// CHECK: store i64 ptrtoint (ptr @fn to i64), ptr [[TMP]], align 8
// CHECK: ret i64 %{{.*}}
