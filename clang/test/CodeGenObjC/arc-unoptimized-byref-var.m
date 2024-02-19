// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -o - %s | FileCheck -check-prefix=CHECK-UNOPT %s

void test19(void) {
   __block id x;
   ^{ (void)x; };
// CHECK-UNOPT-LABEL: define internal void @__Block_byref_object_copy
// CHECK-UNOPT: [[X:%.*]] = getelementptr inbounds [[BYREF_T:%.*]], ptr [[VAR:%.*]], i32 0, i32 6
// CHECK-UNOPT: [[X2:%.*]] = getelementptr inbounds [[BYREF_T:%.*]], ptr [[VAR1:%.*]], i32 0, i32 6
// CHECK-UNOPT-NEXT: [[SIX:%.*]] = load ptr, ptr [[X2]], align 8
// CHECK-UNOPT-NEXT: store ptr null, ptr [[X]], align 8
// CHECK-UNOPT-NEXT: call void @llvm.objc.storeStrong(ptr [[X]], ptr [[SIX]]) [[NUW:#[0-9]+]]
// CHECK-UNOPT-NEXT: call void @llvm.objc.storeStrong(ptr [[X2]], ptr null) [[NUW]]
// CHECK-UNOPT-NEXT: ret void
}

// CHECK: attributes [[NUW]] = { nounwind }
