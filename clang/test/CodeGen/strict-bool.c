// RUN: %clang_cc1 -triple armv7-apple-darwin -O1 -fstrict-bool -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK-STRICT
// RUN: %clang_cc1 -triple armv7-apple-darwin -O1 -fno-strict-bool -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK-NO-STRICT
// RUN: %clang -target armv7-apple-darwin -O1 -mkernel -S -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK-NO-STRICT

struct has_bool {
    _Bool b;
};

int foo(struct has_bool *b) {
    // CHECK-STRICT: load i8, {{.*}}, !range ![[RANGE_BOOL:[0-9]+]]
    // CHECK-STRICT-NOT: and i8

    // CHECK-NO-STRICT: [[BOOL:%.+]] = load i8
    // CHECK-NO-STRICT: and i8 [[BOOL]], 1
    return b->b;
}

// CHECK_STRICT: ![[RANGE_BOOL]] = !{i8 0, i8 2}
