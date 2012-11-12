// We run this twice, once as Objective-C and once as Objective-C++.
// RUN: %clang_cc1 %s -emit-llvm -o - -fobjc-gc -fblocks -fexceptions -triple i386-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 | FileCheck %s
// RUN: %clang_cc1 %s -emit-llvm -o - -fobjc-gc -fblocks -fexceptions -triple i386-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 -x objective-c++ | FileCheck %s


// CHECK: define i8* @{{.*}}test0
// CHECK: define internal void @{{.*}}_block_invoke(
// CHECK:      call i8* @objc_assign_strongCast(
// CHECK-NEXT: ret void
id test0(id x) {
  __block id result;
  ^{ result = x; }();
  return result;
}

// <rdar://problem/8224178>: cleanup __block variables on EH path
// CHECK: define void @{{.*}}test1
void test1() {
  extern void test1_help(void (^x)(void));

  // CHECK:      [[N:%.*]] = alloca [[N_T:%.*]], align 8
  // CHECK:      [[T0:%.*]] = getelementptr inbounds [[N_T]]* [[N]], i32 0, i32 4
  // CHECK-NEXT: store double 1.000000e+{{0?}}01, double* [[T0]], align 8
  __block double n = 10;

  // CHECK:      invoke void @{{.*}}test1_help
  test1_help(^{ n = 20; });

  // CHECK:      [[T1:%.*]] = bitcast [[N_T]]* [[N]] to i8*
  // CHECK-NEXT: call void @_Block_object_dispose(i8* [[T1]], i32 8)
  // CHECK-NEXT: ret void

  // CHECK:      landingpad { i8*, i32 } personality
  // CHECK-NEXT:   cleanup
  // CHECK:      [[T1:%.*]] = bitcast [[N_T]]* [[N]] to i8*
  // CHECK-NEXT: call void @_Block_object_dispose(i8* [[T1]], i32 8)
  // CHECK:      resume { i8*, i32 }
}
