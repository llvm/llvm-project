// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

long fn(void) { return __builtin_bit_cast(long, (long)&fn); }

// CHECK-LABEL: define{{.*}} i64 @fn()
// CHECK: store i64 ptrtoint (ptr @fn to i64), ptr %ref.tmp, align 8
// CHECK: ret i64 %{{.*}}
