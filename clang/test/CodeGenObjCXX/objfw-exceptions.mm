// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-llvm -fexceptions -fobjc-exceptions -fobjc-runtime=objfw -fcxx-exceptions -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-DWARF
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-llvm -fexceptions -fobjc-exceptions -fobjc-runtime=objfw -fcxx-exceptions -exception-model=sjlj -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-SJLJ

@interface OCType @end
void opaque();

// CHECK: define{{.*}} void @_Z3foov()
// CHECK-DWARF-SAME: personality i8* bitcast (i32 (...)* @__gnu_objc_personality_v0 to i8*)
// CHECK-SJLJ-SAME: personality i8* bitcast (i32 (...)* @__gnu_objc_personality_sj0 to i8*)
void foo() {
try {
// CHECK: invoke void @_Z6opaquev
opaque();
} catch (OCType *T) {
// CHECK:      landingpad { i8*, i32 }
}
}
