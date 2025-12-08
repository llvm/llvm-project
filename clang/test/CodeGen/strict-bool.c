// RUN: %clang_cc1 -triple armv7-apple-darwin -O1 -emit-llvm -o - %s | FileCheck %s -check-prefixes=CHECK,CHECK-STRICT
// RUN: %clang_cc1 -triple armv7-apple-darwin -O1 -load-bool-from-mem=strict -emit-llvm -o - %s | FileCheck %s -check-prefixes=CHECK,CHECK-STRICT
// RUN: %clang_cc1 -triple armv7-apple-darwin -O1 -load-bool-from-mem=nonstrict -emit-llvm -o - %s | FileCheck %s -check-prefixes=CHECK,CHECK-NONZERO
// RUN: %clang_cc1 -triple armv7-apple-darwin -O1 -load-bool-from-mem=truncate -emit-llvm -o - %s | FileCheck %s -check-prefixes=CHECK,CHECK-TRUNCATE
// RUN: %clang_cc1 -triple armv7-apple-darwin -O1 -load-bool-from-mem=nonzero -emit-llvm -o - %s | FileCheck %s -check-prefixes=CHECK,CHECK-NONZERO
// RUN: %clang_cc1 -triple armv7-apple-darwin -O1 -fsanitize=bool -load-bool-from-mem=strict -emit-llvm -o - %s | FileCheck %s -check-prefixes=CHECK,CHECK-UBSAN-STRICT
// RUN: %clang_cc1 -triple armv7-apple-darwin -O1 -fsanitize=bool -load-bool-from-mem=truncate -emit-llvm -o - %s | FileCheck %s -check-prefixes=CHECK,CHECK-UBSAN-TRUNCATE

struct has_bool {
    _Bool b;
    unsigned _BitInt(1) c;
};

// CHECK: @foo
int foo(struct has_bool *b) {
    // CHECK-STRICT: [[BOOL:%.+]] = load i8, ptr {{.+}}, !range ![[RANGE_BOOL:[0-9]+]]
    // CHECK-STRICT-NOT: and i8 [[BOOL]], 1
    // CHECK-STRICT-NOT: icmp ne i8 [[BOOL]], 0

    // CHECK-TRUNCATE-NOT: !range
    // CHECK-TRUNCATE: [[BOOL:%.+]] = load i8
    // CHECK-TRUNCATE: and i8 [[BOOL]], 1

    // CHECK-NONZERO-NOT: !range
    // CHECK-NONZERO: [[BOOL:%.+]] = load i8
    // CHECK-NONZERO: icmp ne i8 [[BOOL]], 0

    // CHECK-UBSAN-STRICT-NOT: !range
    // CHECK-UBSAN-STRICT: [[BOOL:%.+]] = load i8, ptr {{.+}}
    // CHECK-UBSAN-STRICT: icmp ult i8 [[BOOL]], 2

    // CHECK-UBSAN-TRUNCATE-NOT: !range
    // CHECK-UBSAN-TRUNCATE: [[BOOL:%.+]] = load i8, ptr {{.+}}
    // CHECK-UBSAN-TRUNCATE: icmp ult i8 [[BOOL]], 2
    return b->b;
}

// CHECK: @bar
int bar(struct has_bool *c) {
    // CHECK-STRICT: [[BITINT:%.+]] = load i8, ptr {{.+}}, !range ![[RANGE_BOOL:[0-9]+]]
    // CHECK-STRICT-NOT: and i8 [[BITINT]], 1
    // CHECK-STRICT-NOT: icmp ne i8 [[BITINT]], 0

    // CHECK-TRUNCATE-NOT: !range
    // CHECK-TRUNCATE: [[BITINT:%.+]] = load i8
    // CHECK-TRUNCATE: and i8 [[BITINT]], 1

    // CHECK-NONZERO-NOT: !range
    // CHECK-NONZERO: [[BITINT:%.+]] = load i8
    // CHECK-NONZERO: icmp ne i8 [[BITINT]], 0

    // CHECK-UBSAN-STRICT-NOT: !range
    // CHECK-UBSAN-STRICT: [[BITINT:%.+]] = load i8, ptr {{.+}}
    // CHECK-UBSAN-STRICT: icmp ult i8 [[BITINT]], 2

    // CHECK-UBSAN-TRUNCATE-NOT: !range
    // CHECK-UBSAN-TRUNCATE: [[BITINT:%.+]] = load i8, ptr {{.+}}
    // CHECK-UBSAN-TRUNCATE: icmp ult i8 [[BITINT]], 2
    return c->c;
}

// CHECK_STRICT: ![[RANGE_BOOL]] = !{i8 0, i8 2}
