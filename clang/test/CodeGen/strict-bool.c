// RUN: %clang_cc1 -triple armv7-apple-darwin -O1 -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK-STRICT
// RUN: %clang_cc1 -triple armv7-apple-darwin -O1 -load-bool-from-mem=strict -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK-STRICT
// RUN: %clang_cc1 -triple armv7-apple-darwin -O1 -load-bool-from-mem=nonstrict -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK-TRUNCATE
// RUN: %clang_cc1 -triple armv7-apple-darwin -O1 -load-bool-from-mem=truncate -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK-TRUNCATE
// RUN: %clang_cc1 -triple armv7-apple-darwin -O1 -load-bool-from-mem=nonzero -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK-NONZERO

struct has_bool {
    _Bool b;
};

int foo(struct has_bool *b) {
    // CHECK-STRICT: load i8, {{.*}}, !range ![[RANGE_BOOL:[0-9]+]]
    // CHECK-STRICT-NOT: and i8

    // CHECK-TRUNCATE: [[BOOL:%.+]] = load i8
    // CHECK-TRUNCATE: and i8 [[BOOL]], 1

    // CHECK-NONZERO: [[BOOL:%.+]] = load i8
    // CHECK-NONZERO: cmp ne i8 [[BOOL]], 0
    return b->b;
}

// CHECK_STRICT: ![[RANGE_BOOL]] = !{i8 0, i8 2}
