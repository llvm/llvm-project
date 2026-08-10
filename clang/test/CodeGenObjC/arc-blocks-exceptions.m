// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -fexceptions -disable-llvm-passes -o - %s | FileCheck %s

void test1(_Bool c) {
  void test1_fn(void (^blk)(void));
  __weak id weakId = 0;
  test1_fn(c ? ^{ (void)weakId; } : 0);

  // CHECK: [[CLEANUP_SAVE:%cond-cleanup.save.*]] = alloca ptr
  // CHECK-NEXT: [[CLEANUP_COND:%.*]] = alloca i1
  // CHECK-NEXT: [[CLEANUP_COND1:%.*]] = alloca i1

  // CHECK: store i1 false, ptr [[CLEANUP_COND]]
  // CHECK-NEXT: store i1 false, ptr [[CLEANUP_COND1]]

  // CHECK: store ptr {{.*}}, ptr [[CLEANUP_SAVE]]
  // CHECK-NEXT: store i1 true, ptr [[CLEANUP_COND]]
  // CHECK-NEXT: store i1 true, ptr [[CLEANUP_COND1]]

  // CHECK: invoke void @test1_fn(
  // CHECK-NEXT: to label %[[INVOKE_CONT:.*]] unwind label %[[LANDING_PAD_LAB:.*]]

  // CHECK: [[INVOKE_CONT]]:
  // CHECK-NEXT: [[LOAD:%.*]] = load i1, ptr [[CLEANUP_COND1]]
  // CHECK-NEXT: br i1 [[LOAD]], label %[[END_OF_SCOPE_LAB:.*]], label

  // CHECK: [[END_OF_SCOPE_LAB]]:
  // CHECK-NEXT: [[LOAD:%.*]] = load ptr, ptr [[CLEANUP_SAVE]]
  // CHECK-NEXT: call void @llvm.objc.destroyWeak(ptr [[LOAD]])
  // CHECK-NEXT: br label

  // CHECK: [[LANDING_PAD_LAB]]:
  //   /* some EH stuff */
  // CHECK:      [[LOAD:%.*]] = load i1, ptr [[CLEANUP_COND]]
  // CHECK-NEXT: br i1 [[LOAD]], label %[[EH_CLEANUP_LAB:.*]], label

  // CHECK: [[EH_CLEANUP_LAB]]:
  // CHECK-NEXT: [[LOAD:%.*]] = load ptr, ptr [[CLEANUP_SAVE]]
  // CHECK-NEXT: call void @llvm.objc.destroyWeak(ptr [[LOAD]])
  // CHECK-NEXT: br label
}
