// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -O2 -disable-llvm-passes -o - %s | FileCheck %s

id g0, g1;

void test0(_Bool cond) {
  id test0_helper(void) __attribute__((ns_returns_retained));

  // CHECK-LABEL:      define{{.*}} void @test0(
  // CHECK:      [[COND:%.*]] = alloca i8,
  // CHECK-NEXT: [[X:%.*]] = alloca ptr,
  // CHECK-NEXT: [[RELVAL:%.*]] = alloca ptr
  // CHECK-NEXT: [[RELCOND:%.*]] = alloca i1
  // CHECK-NEXT: zext
  // CHECK-NEXT: store
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(ptr [[X]])
  // CHECK-NEXT: [[T0:%.*]] = load i8, ptr [[COND]]
  // CHECK-NEXT: [[T1:%.*]] = trunc i8 [[T0]] to i1
  // CHECK-NEXT: store i1 false, ptr [[RELCOND]]
  // CHECK-NEXT: br i1 [[T1]],
  // CHECK:      br label
  // CHECK:      [[CALL:%.*]] = call ptr @test0_helper()
  // CHECK-NEXT: store ptr [[CALL]], ptr [[RELVAL]]
  // CHECK-NEXT: store i1 true, ptr [[RELCOND]]
  // CHECK-NEXT: br label
  // CHECK:      [[T0:%.*]] = phi ptr [ null, {{%.*}} ], [ [[CALL]], {{%.*}} ]
  // CHECK-NEXT: [[T1:%.*]] = call ptr @llvm.objc.retain(ptr [[T0]]) [[NUW:#[0-9]+]]
  // CHECK-NEXT: store ptr [[T1]], ptr [[X]],
  // CHECK-NEXT: [[REL:%.*]] = load i1, ptr [[RELCOND]]
  // CHECK-NEXT: br i1 [[REL]],
  // CHECK:      [[T0:%.*]] = load ptr, ptr [[RELVAL]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]]) [[NUW]]
  // CHECK-NEXT: br label
  // CHECK:      [[T0:%.*]] = load ptr, ptr [[X]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]]) [[NUW]]
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(ptr [[X]])
  // CHECK-NEXT: ret void
  id x = (cond ? 0 : test0_helper());
}

void test1(int cond) {
  __strong id strong;
  __weak id weak;
  extern void test1_sink(id *);
  test1_sink(cond ? &strong : 0);
  test1_sink(cond ? &weak : 0);

  // CHECK-LABEL:    define{{.*}} void @test1(
  // CHECK:      [[COND:%.*]] = alloca i32
  // CHECK-NEXT: [[STRONG:%.*]] = alloca ptr
  // CHECK-NEXT: [[WEAK:%.*]] = alloca ptr
  // CHECK-NEXT: [[TEMP1:%.*]] = alloca ptr
  // CHECK-NEXT: [[TEMP2:%.*]] = alloca ptr
  // CHECK-NEXT: [[CONDCLEANUPSAVE:%.*]] = alloca ptr
  // CHECK-NEXT: [[CONDCLEANUP:%.*]] = alloca i1
  // CHECK-NEXT: store i32
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(ptr [[STRONG]])
  // CHECK-NEXT: store ptr null, ptr [[STRONG]]
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(ptr [[WEAK]])
  // CHECK-NEXT: call ptr @llvm.objc.initWeak(ptr [[WEAK]], ptr null)

  // CHECK-NEXT: [[T0:%.*]] = load i32, ptr [[COND]]
  // CHECK-NEXT: [[T1:%.*]] = icmp ne i32 [[T0]], 0
  // CHECK:      [[ARG:%.*]] = phi ptr
  // CHECK-NEXT: [[T0:%.*]] = icmp eq ptr [[ARG]], null
  // CHECK-NEXT: [[T1:%.*]] = select i1 [[T0]], ptr null, ptr [[TEMP1]]
  // CHECK-NEXT: br i1 [[T0]],
  // CHECK:      [[T0:%.*]] = load ptr, ptr [[ARG]]
  // CHECK-NEXT: store ptr [[T0]], ptr [[TEMP1]]
  // CHECK-NEXT: br label
  // CHECK:      [[W:%.*]] = phi ptr [ [[T0]], {{%.*}} ], [ poison, {{%.*}} ]
  // CHECK-NEXT: call void @test1_sink(ptr noundef [[T1]])
  // CHECK-NEXT: [[T0:%.*]] = icmp eq ptr [[ARG]], null
  // CHECK-NEXT: br i1 [[T0]],
  // CHECK:      [[T0:%.*]] = load ptr, ptr [[TEMP1]]
  // CHECK-NEXT: [[T1:%.*]] = call ptr @llvm.objc.retain(ptr [[T0]])
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.use(ptr [[W]]) [[NUW]]
  // CHECK-NEXT: [[T2:%.*]] = load ptr, ptr [[ARG]]
  // CHECK-NEXT: store ptr [[T1]], ptr [[ARG]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T2]])
  // CHECK-NEXT: br label

  // CHECK:      [[T0:%.*]] = load i32, ptr [[COND]]
  // CHECK-NEXT: [[T1:%.*]] = icmp ne i32 [[T0]], 0
  // CHECK:      [[ARG:%.*]] = phi ptr
  // CHECK-NEXT: [[T0:%.*]] = icmp eq ptr [[ARG]], null
  // CHECK-NEXT: [[T1:%.*]] = select i1 [[T0]], ptr null, ptr [[TEMP2]]
  // CHECK-NEXT: store i1 false, ptr [[CONDCLEANUP]]
  // CHECK-NEXT: br i1 [[T0]],
  // CHECK:      [[T0:%.*]] = call ptr @llvm.objc.loadWeakRetained(ptr [[ARG]])
  // CHECK-NEXT: store ptr [[T0]], ptr [[CONDCLEANUPSAVE]]
  // CHECK-NEXT: store i1 true, ptr [[CONDCLEANUP]]
  // CHECK-NEXT: store ptr [[T0]], ptr [[TEMP2]]
  // CHECK-NEXT: br label
  // CHECK:      call void @test1_sink(ptr noundef [[T1]])
  // CHECK-NEXT: [[T0:%.*]] = icmp eq ptr [[ARG]], null
  // CHECK-NEXT: br i1 [[T0]],
  // CHECK:      [[T0:%.*]] = load ptr, ptr [[TEMP2]]
  // CHECK-NEXT: call ptr @llvm.objc.storeWeak(ptr [[ARG]], ptr [[T0]])
  // CHECK-NEXT: br label

  // CHECK:      call void @llvm.objc.destroyWeak(ptr [[WEAK]])
  // CHECK:      call void @llvm.lifetime.end.p0(ptr [[WEAK]])
  // CHECK:      call void @llvm.lifetime.end.p0(ptr [[STRONG]])
  // CHECK:      ret void
}

// Test that, when emitting an expression at +1 that we can't peephole,
// we emit the retain inside the full-expression.  If we ever peephole
// +1s of conditional expressions (which we probably ought to), we'll
// need to find another example of something we need to do this for.
void test2(int cond) {
  extern id test2_producer(void);
  for (id obj in cond ? test2_producer() : (void*) 0) {
  }

  // CHECK-LABEL:    define{{.*}} void @test2(
  // CHECK:      [[COND:%.*]] = alloca i32,
  // CHECK:      alloca ptr
  // CHECK:      [[CLEANUP_SAVE:%.*]] = alloca ptr
  // CHECK:      [[RUN_CLEANUP:%.*]] = alloca i1
  //   Evaluate condition; cleanup disabled by default.
  // CHECK:      [[T0:%.*]] = load i32, ptr [[COND]],
  // CHECK-NEXT: icmp ne i32 [[T0]], 0
  // CHECK-NEXT: store i1 false, ptr [[RUN_CLEANUP]]
  // CHECK-NEXT: br i1
  //   Within true branch, cleanup enabled.
  // CHECK:      [[T1:%.*]] = call ptr @test2_producer() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T1]])
  // CHECK-NEXT: store ptr [[T1]], ptr [[CLEANUP_SAVE]]
  // CHECK-NEXT: store i1 true, ptr [[RUN_CLEANUP]]
  // CHECK-NEXT: br label
  //   Join point for conditional operator; retain immediately.
  // CHECK:      [[T0:%.*]] = phi ptr [ [[T1]], {{%.*}} ], [ null, {{%.*}} ]
  // CHECK-NEXT: [[RESULT:%.*]] = call ptr @llvm.objc.retain(ptr [[T0]])
  //   Leaving full-expression; run conditional cleanup.
  // CHECK-NEXT: [[T0:%.*]] = load i1, ptr [[RUN_CLEANUP]]
  // CHECK-NEXT: br i1 [[T0]]
  // CHECK:      [[T0:%.*]] = load ptr, ptr [[CLEANUP_SAVE]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]])
  // CHECK-NEXT: br label
  //   And way down at the end of the loop:
  // CHECK:      call void @llvm.objc.release(ptr [[RESULT]])
}

void test3(int cond) {
  __strong id *p = cond ? (__strong id[]){g0, g1} : (__strong id[]){g1, g0};
  test2(cond);

  // CHECK: define{{.*}} void @test3(
  // CHECK: %[[P:.*]] = alloca ptr, align 8
  // CHECK: %[[_COMPOUNDLITERAL:.*]] = alloca [2 x ptr], align 8
  // CHECK: %[[CLEANUP_COND:.*]] = alloca i1, align 1
  // CHECK: %[[_COMPOUNDLITERAL1:.*]] = alloca [2 x ptr], align 8
  // CHECK: %[[CLEANUP_COND4:.*]] = alloca i1, align 1

  // CHECK: %[[V2:.*]] = load ptr, ptr @g0, align 8
  // CHECK: %[[V3:.*]] = call ptr @llvm.objc.retain(ptr %[[V2]])
  // CHECK: store ptr %[[V3]], ptr %[[_COMPOUNDLITERAL]], align 8
  // CHECK: %[[ARRAYINIT_ELEMENT:.*]] = getelementptr inbounds ptr, ptr %[[_COMPOUNDLITERAL]], i64 1
  // CHECK: %[[V4:.*]] = load ptr, ptr @g1, align 8
  // CHECK: %[[V5:.*]] = call ptr @llvm.objc.retain(ptr %[[V4]])
  // CHECK: store ptr %[[V5]], ptr %[[ARRAYINIT_ELEMENT]], align 8
  // CHECK: store i1 true, ptr %[[CLEANUP_COND]], align 1
  // CHECK: %[[ARRAYDECAY:.*]] = getelementptr inbounds [2 x ptr], ptr %[[_COMPOUNDLITERAL]], i64 0, i64 0

  // CHECK: %[[V6:.*]] = load ptr, ptr @g1, align 8
  // CHECK: %[[V7:.*]] = call ptr @llvm.objc.retain(ptr %[[V6]])
  // CHECK: store ptr %[[V7]], ptr %[[_COMPOUNDLITERAL1]], align 8
  // CHECK: %[[ARRAYINIT_ELEMENT3:.*]] = getelementptr inbounds ptr, ptr %[[_COMPOUNDLITERAL1]], i64 1
  // CHECK: %[[V8:.*]] = load ptr, ptr @g0, align 8
  // CHECK: %[[V9:.*]] = call ptr @llvm.objc.retain(ptr %[[V8]])
  // CHECK: store ptr %[[V9]], ptr %[[ARRAYINIT_ELEMENT3]], align 8
  // CHECK: store i1 true, ptr %[[CLEANUP_COND4]], align 1
  // CHECK: %[[ARRAYDECAY5:.*]] = getelementptr inbounds [2 x ptr], ptr %[[_COMPOUNDLITERAL1]], i64 0, i64 0

  // CHECK: %[[COND6:.*]] = phi ptr [ %[[ARRAYDECAY]], %{{.*}} ], [ %[[ARRAYDECAY5]], %{{.*}} ]
  // CHECK: store ptr %[[COND6]], ptr %[[P]], align 8
  // CHECK: call void @test2(

  // CHECK: %[[ARRAY_BEGIN:.*]] = getelementptr inbounds [2 x ptr], ptr %[[_COMPOUNDLITERAL1]], i32 0, i32 0
  // CHECK: %[[V11:.*]] = getelementptr inbounds ptr, ptr %[[ARRAY_BEGIN]], i64 2

  // CHECK: %[[ARRAYDESTROY_ELEMENTPAST:.*]] = phi ptr [ %[[V11]], %{{.*}} ], [ %[[ARRAYDESTROY_ELEMENT:.*]], %{{.*}} ]
  // CHECK: %[[ARRAYDESTROY_ELEMENT]] = getelementptr inbounds ptr, ptr %[[ARRAYDESTROY_ELEMENTPAST]], i64 -1
  // CHECK: %[[V12:.*]] = load ptr, ptr %[[ARRAYDESTROY_ELEMENT]], align 8
  // CHECK: call void @llvm.objc.release(ptr %[[V12]])

  // CHECK: %[[ARRAY_BEGIN10:.*]] = getelementptr inbounds [2 x ptr], ptr %[[_COMPOUNDLITERAL]], i32 0, i32 0
  // CHECK: %[[V13:.*]] = getelementptr inbounds ptr, ptr %[[ARRAY_BEGIN10]], i64 2

  // CHECK: %[[ARRAYDESTROY_ELEMENTPAST12:.*]] = phi ptr [ %[[V13]], %{{.*}} ], [ %[[ARRAYDESTROY_ELEMENT13:.*]], %{{.*}} ]
  // CHECK: %[[ARRAYDESTROY_ELEMENT13]] = getelementptr inbounds ptr, ptr %[[ARRAYDESTROY_ELEMENTPAST12]], i64 -1
  // CHECK: %[[V14:.*]] = load ptr, ptr %[[ARRAYDESTROY_ELEMENT13]], align 8
  // CHECK: call void @llvm.objc.release(ptr %[[V14]])
}

// CHECK: attributes [[NUW]] = { nounwind }
