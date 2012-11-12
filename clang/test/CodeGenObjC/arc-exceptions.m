// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-arc -fexceptions -fobjc-exceptions -fobjc-runtime-has-weak -o - %s | FileCheck %s

@class Ety;

// These first two tests are all PR11732 / rdar://problem/10667070.

void test0_helper(void);
void test0(void) {
  @try {
    test0_helper();
  } @catch (Ety *e) {
  }
}
// CHECK: define void @test0()
// CHECK:      [[E:%.*]] = alloca [[ETY:%.*]]*, align 8
// CHECK-NEXT: invoke void @test0_helper()
// CHECK:      [[T0:%.*]] = call i8* @objc_begin_catch(
// CHECK-NEXT: [[T1:%.*]] = bitcast i8* [[T0]] to [[ETY]]*
// CHECK-NEXT: [[T2:%.*]] = bitcast [[ETY]]* [[T1]] to i8*
// CHECK-NEXT: [[T3:%.*]] = call i8* @objc_retain(i8* [[T2]]) nounwind
// CHECK-NEXT: [[T4:%.*]] = bitcast i8* [[T3]] to [[ETY]]*
// CHECK-NEXT: store [[ETY]]* [[T4]], [[ETY]]** [[E]]
// CHECK-NEXT: [[T0:%.*]] = bitcast [[ETY]]** [[E]] to i8**
// CHECK-NEXT: call void @objc_storeStrong(i8** [[T0]], i8* null) nounwind
// CHECK-NEXT: call void @objc_end_catch() nounwind

void test1_helper(void);
void test1(void) {
  @try {
    test1_helper();
  } @catch (__weak Ety *e) {
  }
}
// CHECK: define void @test1()
// CHECK:      [[E:%.*]] = alloca [[ETY:%.*]]*, align 8
// CHECK-NEXT: invoke void @test1_helper()
// CHECK:      [[T0:%.*]] = call i8* @objc_begin_catch(
// CHECK-NEXT: [[T1:%.*]] = bitcast i8* [[T0]] to [[ETY]]*
// CHECK-NEXT: [[T2:%.*]] = bitcast [[ETY]]** [[E]] to i8**
// CHECK-NEXT: [[T3:%.*]] = bitcast [[ETY]]* [[T1]] to i8*
// CHECK-NEXT: call i8* @objc_initWeak(i8** [[T2]], i8* [[T3]]) nounwind
// CHECK-NEXT: [[T0:%.*]] = bitcast [[ETY]]** [[E]] to i8**
// CHECK-NEXT: call void @objc_destroyWeak(i8** [[T0]]) nounwind
// CHECK-NEXT: call void @objc_end_catch() nounwind
