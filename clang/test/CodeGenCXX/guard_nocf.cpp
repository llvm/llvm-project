// RUN: %clang_cc1 -triple %ms_abi_triple -fms-extensions -std=c++11 -emit-llvm -O2 -o - %s | FileCheck %s

void target_func();
void (*func_ptr)() = &target_func;

// The "guard_nocf" attribute must be added.
__declspec(guard(nocf)) void nocf0() {
  (*func_ptr)();
}
// CHECK-LABEL: nocf0
// CHECK: call{{.*}}[[NOCF:#[0-9]+]]

// The "guard_nocf" attribute must *not* be added.
void cf0() {
  (*func_ptr)();
}
// CHECK-LABEL: cf0
// CHECK: call{{.*}}[[CF:#[0-9]+]]

// If the modifier is present on either the function declaration or definition,
// the "guard_nocf" attribute must be added.
__declspec(guard(nocf)) void nocf1();
void nocf1() {
  (*func_ptr)();
}
// CHECK-LABEL: nocf1
// CHECK: call{{.*}}[[NOCF:#[0-9]+]]

void nocf2();
__declspec(guard(nocf)) void nocf2() {
  (*func_ptr)();
}
// CHECK-LABEL: nocf2
// CHECK: call{{.*}}[[NOCF:#[0-9]+]]

// When inlining a function, the "guard_nocf" attribute on indirect calls must
// be preserved.
void nocf3() {
  nocf0();
}
// CHECK-LABEL: nocf3
// CHECK: call{{.*}}[[NOCF:#[0-9]+]]

// When inlining into a function marked as __declspec(guard(nocf)), the
// "guard_nocf" attribute must *not* be added to the inlined calls.
__declspec(guard(nocf)) void cf1() {
  cf0();
}
// CHECK-LABEL: cf1
// CHECK: call{{.*}}[[CF:#[0-9]+]]

// When the __declspec(guard(nocf)) modifier is present on an override function,
// the "guard_nocf" attribute must be added.
struct Base {
  virtual void nocf4();
};

struct Derived : Base {
  __declspec(guard(nocf)) void nocf4() override {
    (*func_ptr)();
  }
};
Derived d;
// CHECK-LABEL: nocf4
// CHECK: call{{.*}}[[NOCF:#[0-9]+]]

// When the modifier is not present on an override function, the "guard_nocf"
// attribute must *not* be added, even if the modifier is present on the virtual
// function.
struct Base1 {
  __declspec(guard(nocf)) virtual void cf2();
};

struct Derived1 : Base1 {
  void cf2() override {
    (*func_ptr)();
  }
};
Derived1 d1;
// CHECK-LABEL: cf2
// CHECK: call{{.*}}[[CF:#[0-9]+]]

// CHECK: attributes [[NOCF]] = { {{.*}}"guard_nocf"{{.*}} }
// CHECK-NOT: attributes [[CF]] = { {{.*}}"guard_nocf"{{.*}} }
