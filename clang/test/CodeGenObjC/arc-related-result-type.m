// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fblocks -fobjc-arc -o - %s | FileCheck %s

@interface Test0
- (id) self;
@end
void test0(Test0 *val) {
  Test0 *x = [val self];

// CHECK-LABEL:    define{{.*}} void @test0(
// CHECK:      [[VAL:%.*]] = alloca [[TEST0:%.*]]*
// CHECK-NEXT: [[X:%.*]] = alloca [[TEST0]]*
// CHECK-NEXT: store [[TEST0]]* null
// CHECK-NEXT: bitcast
// CHECK-NEXT: bitcast
// CHECK-NEXT: call void @llvm.objc.storeStrong(
// CHECK-NEXT: load [[TEST0]]*, [[TEST0]]** [[VAL]],
// CHECK-NEXT: load
// CHECK-NEXT: bitcast
// CHECK-NEXT: [[T0:%.*]] = call i8* bitcast (
// CHECK-NEXT: [[T1:%.*]] = notail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* [[T0]])
// CHECK-NEXT: [[T2:%.*]] = bitcast i8* [[T1]] to [[TEST0]]*
// CHECK-NEXT: store [[TEST0]]* [[T2]], [[TEST0]]** [[X]]
// CHECK-NEXT: [[T0:%.*]] = bitcast [[TEST0]]** [[X]] to i8**
// CHECK-NEXT: call void @llvm.objc.storeStrong(i8** [[T0]], i8* null)
// CHECK-NEXT: [[T0:%.*]] = bitcast [[TEST0]]** [[VAL]] to i8**
// CHECK-NEXT: call void @llvm.objc.storeStrong(i8** [[T0]], i8* null)
// CHECK-NEXT: ret void
}
