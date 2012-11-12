// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-arc -fexceptions -fobjc-exceptions -fcxx-exceptions -fobjc-runtime-has-weak -o - %s | FileCheck %s

@class Ety;

// These first four tests are all PR11732 / rdar://problem/10667070.

void test0_helper(void);
void test0(void) {
  @try {
    test0_helper();
  } @catch (Ety *e) {
  }
}
// CHECK: define void @_Z5test0v()
// CHECK:      [[E:%.*]] = alloca [[ETY:%.*]]*, align 8
// CHECK-NEXT: invoke void @_Z12test0_helperv()
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
// CHECK: define void @_Z5test1v()
// CHECK:      [[E:%.*]] = alloca [[ETY:%.*]]*, align 8
// CHECK-NEXT: invoke void @_Z12test1_helperv()
// CHECK:      [[T0:%.*]] = call i8* @objc_begin_catch(
// CHECK-NEXT: [[T1:%.*]] = bitcast i8* [[T0]] to [[ETY]]*
// CHECK-NEXT: [[T2:%.*]] = bitcast [[ETY]]** [[E]] to i8**
// CHECK-NEXT: [[T3:%.*]] = bitcast [[ETY]]* [[T1]] to i8*
// CHECK-NEXT: call i8* @objc_initWeak(i8** [[T2]], i8* [[T3]]) nounwind
// CHECK-NEXT: [[T0:%.*]] = bitcast [[ETY]]** [[E]] to i8**
// CHECK-NEXT: call void @objc_destroyWeak(i8** [[T0]]) nounwind
// CHECK-NEXT: call void @objc_end_catch() nounwind

void test2_helper(void);
void test2(void) {
  try {
    test2_helper();
  } catch (Ety *e) {
  }
}
// CHECK: define void @_Z5test2v()
// CHECK:      [[E:%.*]] = alloca [[ETY:%.*]]*, align 8
// CHECK-NEXT: invoke void @_Z12test2_helperv()
// CHECK:      [[T0:%.*]] = call i8* @__cxa_begin_catch(
// CHECK-NEXT: [[T1:%.*]] = bitcast i8* [[T0]] to [[ETY]]*
// CHECK-NEXT: [[T2:%.*]] = bitcast [[ETY]]* [[T1]] to i8*
// CHECK-NEXT: [[T3:%.*]] = call i8* @objc_retain(i8* [[T2]]) nounwind
// CHECK-NEXT: [[T4:%.*]] = bitcast i8* [[T3]] to [[ETY]]*
// CHECK-NEXT: store [[ETY]]* [[T4]], [[ETY]]** [[E]]
// CHECK-NEXT: [[T0:%.*]] = bitcast [[ETY]]** [[E]] to i8**
// CHECK-NEXT: call void @objc_storeStrong(i8** [[T0]], i8* null) nounwind
// CHECK-NEXT: call void @__cxa_end_catch() nounwind

void test3_helper(void);
void test3(void) {
  try {
    test3_helper();
  } catch (Ety * __weak e) {
  }
}
// CHECK: define void @_Z5test3v()
// CHECK:      [[E:%.*]] = alloca [[ETY:%.*]]*, align 8
// CHECK-NEXT: invoke void @_Z12test3_helperv()
// CHECK:      [[T0:%.*]] = call i8* @__cxa_begin_catch(
// CHECK-NEXT: [[T1:%.*]] = bitcast i8* [[T0]] to [[ETY]]*
// CHECK-NEXT: [[T2:%.*]] = bitcast [[ETY]]** [[E]] to i8**
// CHECK-NEXT: [[T3:%.*]] = bitcast [[ETY]]* [[T1]] to i8*
// CHECK-NEXT: call i8* @objc_initWeak(i8** [[T2]], i8* [[T3]]) nounwind
// CHECK-NEXT: [[T0:%.*]] = bitcast [[ETY]]** [[E]] to i8**
// CHECK-NEXT: call void @objc_destroyWeak(i8** [[T0]]) nounwind
// CHECK-NEXT: call void @__cxa_end_catch() nounwind
