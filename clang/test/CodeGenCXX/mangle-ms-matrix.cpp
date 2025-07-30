// RUN: %clang_cc1 -fenable-matrix -fms-extensions -fcxx-exceptions -ffreestanding -target-feature +avx -emit-llvm %s -o - -triple=i686-pc-win32 | FileCheck %s
// RUN: %clang_cc1 -fenable-matrix -fms-extensions -fcxx-exceptions -ffreestanding -target-feature +avx -emit-llvm %s -o - -triple=i686-pc-win32 -fexperimental-new-constant-interpreter | FileCheck %s

typedef float __attribute__((matrix_type(4, 4))) m4x4f;
typedef float __attribute__((matrix_type(2, 2))) m2x2f;

typedef int __attribute__((matrix_type(4, 4))) m4x4i;
typedef int __attribute__((matrix_type(2, 2))) m2x2i;

void thow(int i) {
  switch (i) {
    case 0: throw m4x4f();
    // CHECK: ??_R0U?$__matrix@M$03$03@__clang@@@8
    // CHECK: _CT??_R0U?$__matrix@M$03$03@__clang@@@864
    // CHECK: _CTA1U?$__matrix@M$03$03@__clang@@
    // CHECK: _TI1U?$__matrix@M$03$03@__clang@@
    case 1: throw m2x2f();
    // CHECK: ??_R0U?$__matrix@M$01$01@__clang@@@8
    // CHECK: _CT??_R0U?$__matrix@M$01$01@__clang@@@816
    // CHECK: _CTA1U?$__matrix@M$01$01@__clang@@
    // CHECK: _TI1U?$__matrix@M$01$01@__clang@@
    case 2: throw m4x4i();
    // CHECK: ??_R0U?$__matrix@H$03$03@__clang@@@8
    // CHECK: _CT??_R0U?$__matrix@H$03$03@__clang@@@864
    // CHECK: _CTA1U?$__matrix@H$03$03@__clang@@
    // CHECK: _TI1U?$__matrix@H$03$03@__clang@@
    case 3: throw m2x2i();
    // CHECK: ??_R0U?$__matrix@H$01$01@__clang@@@8
    // CHECK: _CT??_R0U?$__matrix@H$01$01@__clang@@@816
    // CHECK: _CTA1U?$__matrix@H$01$01@__clang@@
    // CHECK: _TI1U?$__matrix@H$01$01@__clang@@
  }
}

void foo44f(m4x4f) {}
// CHECK: define dso_local void @"?foo44f@@YAXU?$__matrix@M$03$03@__clang@@@Z"

m4x4f rfoo44f() { return m4x4f(); }
// CHECK: define dso_local noundef <16 x float> @"?rfoo44f@@YAU?$__matrix@M$03$03@__clang@@XZ"

void foo22f(m2x2f) {}
// CHECK: define dso_local void @"?foo22f@@YAXU?$__matrix@M$01$01@__clang@@@Z"

m2x2f rfoo22f() { return m2x2f(); }
// CHECK: define dso_local noundef <4 x float> @"?rfoo22f@@YAU?$__matrix@M$01$01@__clang@@XZ"

void foo44i(m4x4i) {}
// CHECK: define dso_local void @"?foo44i@@YAXU?$__matrix@H$03$03@__clang@@@Z"

m4x4i rfoo44i() { return m4x4i(); }
// CHECK: define dso_local noundef <16 x i32> @"?rfoo44i@@YAU?$__matrix@H$03$03@__clang@@XZ"

void foo22i(m2x2i) {}
// CHECK: define dso_local void @"?foo22i@@YAXU?$__matrix@H$01$01@__clang@@@Z"

m2x2i rfoo22i() { return m2x2i(); }
// CHECK: define dso_local noundef <4 x i32> @"?rfoo22i@@YAU?$__matrix@H$01$01@__clang@@XZ"