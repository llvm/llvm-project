// RUN: %clang_cc1 -std=c23 -fdefer-ts -ast-print %s | FileCheck %s

void g();

// CHECK: void f
void f() {
    // CHECK-NEXT: _Defer
    // CHECK-NEXT:     g();
    // CHECK-NEXT: _Defer
    // CHECK-NEXT:     _Defer
    // CHECK-NEXT:         g();
    // CHECK-NEXT: _Defer {
    // CHECK-NEXT: }
    // CHECK-NEXT: _Defer {
    // CHECK-NEXT:     int x;
    // CHECK-NEXT: }
    // CHECK-NEXT: _Defer
    // CHECK-NEXT:     if (1) {
    // CHECK-NEXT:     }
    _Defer
        g();
    _Defer
        _Defer
            g();
    _Defer {
    }
    _Defer {
        int x;
    }
    _Defer
        if (1) {
        }
}
