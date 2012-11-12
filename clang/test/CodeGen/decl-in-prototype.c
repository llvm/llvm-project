// RUN: %clang -target i386-unknown-unknown -emit-llvm -S -o - %s | FileCheck %s

const int AA = 5;

// CHECK: define i32 @f1
int f1(enum {AA,BB} E) {
    // CHECK: ret i32 1
    return BB;
}

// CHECK: define i32 @f2
int f2(enum {AA=7,BB} E) {
    // CHECK: ret i32 7
    return AA;
}

// Check nested function declarators work.
int f(void (*g)(), enum {AA,BB} h) {
    // CHECK: ret i32 0
    return AA;
}
