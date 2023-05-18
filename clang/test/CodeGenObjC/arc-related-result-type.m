// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fblocks -fobjc-arc -o - %s | FileCheck %s

@interface Test0
- (id) self;
@end
void test0(Test0 *val) {
  Test0 *x = [val self];

// CHECK-LABEL:    define{{.*}} void @test0(
// CHECK:      [[VAL:%.*]] = alloca ptr
// CHECK-NEXT: [[X:%.*]] = alloca ptr
// CHECK-NEXT: store ptr null
// CHECK-NEXT: call void @llvm.objc.storeStrong(
// CHECK-NEXT: load ptr, ptr [[VAL]],
// CHECK-NEXT: load
// CHECK-NEXT: [[T0:%.*]] = call ptr
// CHECK-NEXT: [[T1:%.*]] = notail call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr [[T0]])
// CHECK-NEXT: store ptr [[T1]], ptr [[X]]
// CHECK-NEXT: call void @llvm.objc.storeStrong(ptr [[X]], ptr null)
// CHECK-NEXT: call void @llvm.objc.storeStrong(ptr [[VAL]], ptr null)
// CHECK-NEXT: ret void
}
