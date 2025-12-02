// RUN: %clang_cc1 -x c++ -emit-llvm -triple x86_64-unknown-linux-gnu -o - %s | FileCheck --check-prefixes=CHECK,CHECK-ENABLED %s
// RUN: %clang_cc1 -x c++ -emit-llvm -triple x86_64-unknown-linux-gnu -fmark-objects-dead-after-destructors -o - %s | FileCheck --check-prefixes=CHECK,CHECK-ENABLED %s
// RUN: %clang_cc1 -x c++ -emit-llvm -triple x86_64-unknown-linux-gnu -fno-mark-objects-dead-after-destructors -o - %s | FileCheck --check-prefixes=CHECK,CHECK-DISABLED %s

// CHECK: define linkonce_odr void @_ZN3barD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %this)
// CHECK-ENABLED: define linkonce_odr void @_ZN3fooD2Ev(ptr dead_on_return noundef nonnull align 1 dereferenceable(1) %this)
// CHECK-DISABLED: define linkonce_odr void @_ZN3fooD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %this)

int dummyFunction();

class foo {
public:
  ~foo() {
    dummyFunction();
  }
};

class bar : foo {
public:
  ~bar() {
    dummyFunction();
  }
};

int main() {
    bar baz;
    return 0;
}
