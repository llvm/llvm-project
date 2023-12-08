// RUN: %clang_cc1 -triple %ms_abi_triple -fms-extensions -emit-llvm -O2 -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-w64-windows-gnu -emit-llvm -O2 -o - %s | FileCheck %s

// The x86_64-w64-windows-gnu version tests mingw target, which relies on
// __declspec(...) being defined as __attribute__((...)) by compiler built-in.

void target_func(void);
void (*func_ptr)(void) = &target_func;

// The "guard_nocf" attribute must be added.
__declspec(guard(nocf)) void nocf0(void) {
  (*func_ptr)();
}
// CHECK-LABEL: nocf0
// CHECK: call{{.*}}[[NOCF:#[0-9]+]]

// Explicitly test using the GNU-style attribute without relying on the
// __declspec define for mingw.
// The "guard_nocf" attribute must be added.
__attribute__((guard(nocf))) void nocf0_gnu_style(void) {
  (*func_ptr)();
}
// CHECK-LABEL: nocf0_gnu_style
// CHECK: call{{.*}}[[NOCF:#[0-9]+]]

// The "guard_nocf" attribute must *not* be added.
void cf0(void) {
  (*func_ptr)();
}
// CHECK-LABEL: cf0
// CHECK: call{{.*}}[[CF:#[0-9]+]]

// If the modifier is present on either the function declaration or definition,
// the "guard_nocf" attribute must be added.
__declspec(guard(nocf)) void nocf1(void);
void nocf1(void) {
  (*func_ptr)();
}
// CHECK-LABEL: nocf1
// CHECK: call{{.*}}[[NOCF:#[0-9]+]]

void nocf2(void);
__declspec(guard(nocf)) void nocf2(void) {
  (*func_ptr)();
}
// CHECK-LABEL: nocf2
// CHECK: call{{.*}}[[NOCF:#[0-9]+]]

// When inlining a function, the "guard_nocf" attribute on indirect calls must
// be preserved.
void nocf3(void) {
  nocf0();
}
// CHECK-LABEL: nocf3
// CHECK: call{{.*}}[[NOCF:#[0-9]+]]

// When inlining into a function marked as __declspec(guard(nocf)), the
// "guard_nocf" attribute must *not* be added to the inlined calls.
__declspec(guard(nocf)) void cf1(void) {
  cf0();
}
// CHECK-LABEL: cf1
// CHECK: call{{.*}}[[CF:#[0-9]+]]

// CHECK: attributes [[NOCF]] = { {{.*}}"guard_nocf"{{.*}} }
// CHECK-NOT: attributes [[CF]] = { {{.*}}"guard_nocf"{{.*}} }
