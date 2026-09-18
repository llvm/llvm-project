// RUN: %clang_cc1 -triple=x86_64 -emit-llvm -o - %s | FileCheck %s

struct StructArray {
    long a;
    float b;
    struct {
    } c[7];
};

struct UnionArray {
    long a;
    float b;
    union {
    } c[7];
};

// CHECK-LABEL: define{{.*}} @foo_struct
// CHECK: ret
struct StructArray foo_struct() {
    struct StructArray s;
    return s;
}

// CHECK-LABEL: define{{.*}} @foo_union
// CHECK: ret
struct UnionArray foo_union() {
    struct UnionArray s;
    return s;
}
