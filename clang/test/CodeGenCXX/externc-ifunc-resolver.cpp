// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s

extern "C" {
__attribute__((used)) static void *resolve_foo() { return 0; }
__attribute__((ifunc("resolve_foo"))) char *foo();
__attribute__((ifunc("resolve_foo"))) void foo2(int);
__attribute__((ifunc("resolve_foo"))) char foo3(float);
__attribute__((ifunc("resolve_foo"))) char foo4(float);
}

// CHECK: @resolve_foo = internal alias i8* (), i8* ()* @_ZL11resolve_foov
// CHECK: @foo = ifunc i8* (), bitcast (i8* ()* @_ZL11resolve_foov to i8* ()* ()*)
// CHECK: @foo2 = ifunc void (i32), bitcast (i8* ()* @_ZL11resolve_foov to void (i32)* ()*)
// CHECK: @foo3 = ifunc i8 (float), bitcast (i8* ()* @_ZL11resolve_foov to i8 (float)* ()*)
// CHECK: @foo4 = ifunc i8 (float), bitcast (i8* ()* @_ZL11resolve_foov to i8 (float)* ()*)
// CHECK: define internal noundef i8* @_ZL11resolve_foov()
