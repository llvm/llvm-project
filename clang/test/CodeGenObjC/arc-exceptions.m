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
// CHECK-LABEL: define{{.*}} void @test0()
// CHECK:      [[E:%e]] = alloca ptr, align 8
// CHECK:      invoke void @test0_helper()
// CHECK:      [[T0:%.*]] = call ptr @objc_begin_catch(
// CHECK-NEXT: [[T3:%.*]] = call ptr @llvm.objc.retain(ptr [[T0]]) [[NUW:#[0-9]+]]
// CHECK-NEXT: store ptr [[T3]], ptr [[E]]
// CHECK-NEXT: call void @llvm.objc.storeStrong(ptr [[E]], ptr null) [[NUW]]
// CHECK-NEXT: call void @objc_end_catch() [[NUW]]

void test1_helper(void);
void test1(void) {
  @try {
    test1_helper();
  } @catch (__weak Ety *e) {
  }
}
// CHECK-LABEL: define{{.*}} void @test1()
// CHECK:      [[E:%e]] = alloca ptr, align 8
// CHECK:      invoke void @test1_helper()
// CHECK:      [[T0:%.*]] = call ptr @objc_begin_catch(
// CHECK-NEXT: call ptr @llvm.objc.initWeak(ptr [[E]], ptr [[T0]]) [[NUW]]
// CHECK-NEXT: call void @llvm.objc.destroyWeak(ptr [[E]]) [[NUW]]
// CHECK-NEXT: call void @objc_end_catch() [[NUW]]

// CHECK: attributes [[NUW]] = { nounwind }
