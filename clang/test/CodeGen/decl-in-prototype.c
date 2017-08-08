// RUN: %clang_cc1 -triple i386-linux -debug-info-kind=limited -emit-llvm -o - %s | FileCheck %s

const int AA = 5;

// CHECK-LABEL: define i32 @f1
int f1(enum {AA,BB} E) {
    // CHECK: ret i32 1
    return BB;
}

// CHECK-LABEL: define i32 @f2
int f2(enum {AA=7,BB} E) {
    // CHECK: ret i32 7
    return AA;
}

// Check nested function declarators work.
int f(void (*g)(), enum {AA,BB} h) {
    // CHECK: ret i32 0
    return AA;
}

// This used to crash with debug info enabled.
int pr31366(struct { enum { a = 1 } b; } c) {
  return a;
}
