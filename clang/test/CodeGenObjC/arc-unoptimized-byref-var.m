// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -o - %s | FileCheck -check-prefix=CHECK-UNOPT %s
// rdar://12530881

void test19() {
   __block id x;
   ^{ (void)x; };
// CHECK-UNOPT-LABEL: define internal void @__Block_byref_object_copy
// CHECK-UNOPT: [[X:%.*]] = getelementptr inbounds [[BYREF_T:%.*]], [[BYREF_T:%.*]]* [[VAR:%.*]], i32 0, i32 6
// CHECK-UNOPT: [[X2:%.*]] = getelementptr inbounds [[BYREF_T:%.*]], [[BYREF_T:%.*]]* [[VAR1:%.*]], i32 0, i32 6
// CHECK-UNOPT-NEXT: [[SIX:%.*]] = load i8*, i8** [[X2]], align 8
// CHECK-UNOPT-NEXT: store i8* null, i8** [[X]], align 8
// CHECK-UNOPT-NEXT: call void @llvm.objc.storeStrong(i8** [[X]], i8* [[SIX]]) [[NUW:#[0-9]+]]
// CHECK-UNOPT-NEXT: call void @llvm.objc.storeStrong(i8** [[X2]], i8* null) [[NUW]]
// CHECK-UNOPT-NEXT: ret void
}

// CHECK: attributes [[NUW]] = { nounwind }
