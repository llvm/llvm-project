// RUN: %clang_cc1 -std=c23 -fdefer-ts -ast-print %s | FileCheck %s

void g();

// CHECK: void f
void f() {
    // CHECK-NEXT: defer
    // CHECK-NEXT:     g();
    // CHECK-NEXT: defer
    // CHECK-NEXT:     defer
    // CHECK-NEXT:         g();
    // CHECK-NEXT: defer {
    // CHECK-NEXT: }
    // CHECK-NEXT: defer {
    // CHECK-NEXT:     int x;
    // CHECK-NEXT: }
    // CHECK-NEXT: defer
    // CHECK-NEXT:     if (1) {
    // CHECK-NEXT:     }
    defer
        g();
    defer
        defer
            g();
    defer {
    }
    defer {
        int x;
    }
    defer
        if (1) {
        }
}
