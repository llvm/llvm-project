// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-OK

void (__attribute__((regparm(3), stdcall)) *pf) ();
void (__attribute__((regparm(2), stdcall)) foo)(int a) {
}
// CHECK: @pf = common dso_local global void (...)* null
// CHECK: define dso_local void @foo(i32 %a)

// CHECK-OK: @pf = common dso_local global void (...)* null
// CHECK-OK: define dso_local x86_stdcallcc void @foo(i32 inreg %a)
