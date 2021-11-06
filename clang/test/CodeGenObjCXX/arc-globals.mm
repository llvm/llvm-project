// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fblocks -fobjc-arc -O2 -disable-llvm-passes -o - %s | FileCheck %s

// Test that we're properly retaining lifetime-qualified pointers
// initialized statically and wrapping up those initialization in an
// autorelease pool.
id getObject();

// CHECK-LABEL: define internal void @__cxx_global_var_init
// CHECK: call i8* @_Z9getObjectv
// CHECK-NEXT: call i8* @llvm.objc.retainAutoreleasedReturnValue
// CHECK-NEXT: {{store i8*.*@global_obj}}
// CHECK-NEXT: ret void
id global_obj = getObject();

// CHECK-LABEL: define internal void @__cxx_global_var_init
// CHECK: call i8* @_Z9getObjectv
// CHECK-NEXT: call i8* @llvm.objc.retainAutoreleasedReturnValue
// CHECK-NEXT: {{store i8*.*@global_obj2}}
// CHECK-NEXT: ret void
id global_obj2 = getObject();

// CHECK-LABEL: define internal void @_GLOBAL__sub_I_arc_globals.mm
// CHECK: call i8* @llvm.objc.autoreleasePoolPush()
// CHECK-NEXT: call void @__cxx_global_var_init
// CHECK-NEXT: call void @__cxx_global_var_init.1
// CHECK-NEXT: call void @llvm.objc.autoreleasePoolPop(
// CHECK-NEXT: ret void
