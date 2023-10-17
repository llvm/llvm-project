// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s

extern "C" {
__attribute__((used)) static void *resolve_foo() { return 0; }
__attribute__((ifunc("resolve_foo"))) char *foo();
__attribute__((ifunc("resolve_foo"))) void foo2(int);
__attribute__((ifunc("resolve_foo"))) char foo3(float);
__attribute__((ifunc("resolve_foo"))) char foo4(float);
}

// CHECK: @resolve_foo = internal alias ptr (), ptr @_ZL11resolve_foov
// CHECK: @foo = ifunc ptr (), ptr @_ZL11resolve_foov
// CHECK: @foo2 = ifunc void (i32), ptr @_ZL11resolve_foov
// CHECK: @foo3 = ifunc i8 (float), ptr @_ZL11resolve_foov
// CHECK: @foo4 = ifunc i8 (float), ptr @_ZL11resolve_foov
// CHECK: define internal noundef ptr @_ZL11resolve_foov()
