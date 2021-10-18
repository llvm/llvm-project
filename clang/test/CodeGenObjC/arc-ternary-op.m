// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -O2 -disable-llvm-passes -o - %s | FileCheck %s

id g0, g1;

void test0(_Bool cond) {
  id test0_helper(void) __attribute__((ns_returns_retained));

  // CHECK-LABEL:      define{{.*}} void @test0(
  // CHECK:      [[COND:%.*]] = alloca i8,
  // CHECK-NEXT: [[X:%.*]] = alloca i8*,
  // CHECK-NEXT: [[RELVAL:%.*]] = alloca i8*
  // CHECK-NEXT: [[RELCOND:%.*]] = alloca i1
  // CHECK-NEXT: zext
  // CHECK-NEXT: store
  // CHECK-NEXT: [[XPTR1:%.*]] = bitcast i8** [[X]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start.p0i8(i64 8, i8* [[XPTR1]])
  // CHECK-NEXT: [[T0:%.*]] = load i8, i8* [[COND]]
  // CHECK-NEXT: [[T1:%.*]] = trunc i8 [[T0]] to i1
  // CHECK-NEXT: store i1 false, i1* [[RELCOND]]
  // CHECK-NEXT: br i1 [[T1]],
  // CHECK:      br label
  // CHECK:      [[CALL:%.*]] = call i8* @test0_helper()
  // CHECK-NEXT: store i8* [[CALL]], i8** [[RELVAL]]
  // CHECK-NEXT: store i1 true, i1* [[RELCOND]]
  // CHECK-NEXT: br label
  // CHECK:      [[T0:%.*]] = phi i8* [ null, {{%.*}} ], [ [[CALL]], {{%.*}} ]
  // CHECK-NEXT: [[T1:%.*]] = call i8* @llvm.objc.retain(i8* [[T0]]) [[NUW:#[0-9]+]]
  // CHECK-NEXT: store i8* [[T1]], i8** [[X]],
  // CHECK-NEXT: [[REL:%.*]] = load i1, i1* [[RELCOND]]
  // CHECK-NEXT: br i1 [[REL]],
  // CHECK:      [[T0:%.*]] = load i8*, i8** [[RELVAL]]
  // CHECK-NEXT: call void @llvm.objc.release(i8* [[T0]]) [[NUW]]
  // CHECK-NEXT: br label
  // CHECK:      [[T0:%.*]] = load i8*, i8** [[X]]
  // CHECK-NEXT: call void @llvm.objc.release(i8* [[T0]]) [[NUW]]
  // CHECK-NEXT: [[XPTR2:%.*]] = bitcast i8** [[X]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.end.p0i8(i64 8, i8* [[XPTR2]])
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
  // CHECK-NEXT: [[STRONG:%.*]] = alloca i8*
  // CHECK-NEXT: [[WEAK:%.*]] = alloca i8*
  // CHECK-NEXT: [[TEMP1:%.*]] = alloca i8*
  // CHECK-NEXT: [[TEMP2:%.*]] = alloca i8*
  // CHECK-NEXT: [[CONDCLEANUPSAVE:%.*]] = alloca i8*
  // CHECK-NEXT: [[CONDCLEANUP:%.*]] = alloca i1
  // CHECK-NEXT: store i32
  // CHECK-NEXT: [[STRONGPTR1:%.*]] = bitcast i8** [[STRONG]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start.p0i8(i64 8, i8* [[STRONGPTR1]])
  // CHECK-NEXT: store i8* null, i8** [[STRONG]]
  // CHECK-NEXT: [[WEAKPTR1:%.*]] = bitcast i8** [[WEAK]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start.p0i8(i64 8, i8* [[WEAKPTR1]])
  // CHECK-NEXT: call i8* @llvm.objc.initWeak(i8** [[WEAK]], i8* null)

  // CHECK-NEXT: [[T0:%.*]] = load i32, i32* [[COND]]
  // CHECK-NEXT: [[T1:%.*]] = icmp ne i32 [[T0]], 0
  // CHECK:      [[ARG:%.*]] = phi i8**
  // CHECK-NEXT: [[T0:%.*]] = icmp eq i8** [[ARG]], null
  // CHECK-NEXT: [[T1:%.*]] = select i1 [[T0]], i8** null, i8** [[TEMP1]]
  // CHECK-NEXT: br i1 [[T0]],
  // CHECK:      [[T0:%.*]] = load i8*, i8** [[ARG]]
  // CHECK-NEXT: store i8* [[T0]], i8** [[TEMP1]]
  // CHECK-NEXT: br label
  // CHECK:      [[W:%.*]] = phi i8* [ [[T0]], {{%.*}} ], [ undef, {{%.*}} ]
  // CHECK-NEXT: call void @test1_sink(i8** [[T1]])
  // CHECK-NEXT: [[T0:%.*]] = icmp eq i8** [[ARG]], null
  // CHECK-NEXT: br i1 [[T0]],
  // CHECK:      [[T0:%.*]] = load i8*, i8** [[TEMP1]]
  // CHECK-NEXT: [[T1:%.*]] = call i8* @llvm.objc.retain(i8* [[T0]])
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.use(i8* [[W]]) [[NUW]]
  // CHECK-NEXT: [[T2:%.*]] = load i8*, i8** [[ARG]]
  // CHECK-NEXT: store i8* [[T1]], i8** [[ARG]]
  // CHECK-NEXT: call void @llvm.objc.release(i8* [[T2]])
  // CHECK-NEXT: br label

  // CHECK:      [[T0:%.*]] = load i32, i32* [[COND]]
  // CHECK-NEXT: [[T1:%.*]] = icmp ne i32 [[T0]], 0
  // CHECK:      [[ARG:%.*]] = phi i8**
  // CHECK-NEXT: [[T0:%.*]] = icmp eq i8** [[ARG]], null
  // CHECK-NEXT: [[T1:%.*]] = select i1 [[T0]], i8** null, i8** [[TEMP2]]
  // CHECK-NEXT: store i1 false, i1* [[CONDCLEANUP]]
  // CHECK-NEXT: br i1 [[T0]],
  // CHECK:      [[T0:%.*]] = call i8* @llvm.objc.loadWeakRetained(i8** [[ARG]])
  // CHECK-NEXT: store i8* [[T0]], i8** [[CONDCLEANUPSAVE]]
  // CHECK-NEXT: store i1 true, i1* [[CONDCLEANUP]]
  // CHECK-NEXT: store i8* [[T0]], i8** [[TEMP2]]
  // CHECK-NEXT: br label
  // CHECK:      call void @test1_sink(i8** [[T1]])
  // CHECK-NEXT: [[T0:%.*]] = icmp eq i8** [[ARG]], null
  // CHECK-NEXT: br i1 [[T0]],
  // CHECK:      [[T0:%.*]] = load i8*, i8** [[TEMP2]]
  // CHECK-NEXT: call i8* @llvm.objc.storeWeak(i8** [[ARG]], i8* [[T0]])
  // CHECK-NEXT: br label

  // CHECK:      call void @llvm.objc.destroyWeak(i8** [[WEAK]])
  // CHECK:      [[WEAKPTR2:%.*]] = bitcast i8** [[WEAK]] to i8*
  // CHECK:      call void @llvm.lifetime.end.p0i8(i64 8, i8* [[WEAKPTR2]])
  // CHECK:      [[STRONGPTR2:%.*]] = bitcast i8** [[STRONG]] to i8*
  // CHECK:      call void @llvm.lifetime.end.p0i8(i64 8, i8* [[STRONGPTR2]])
  // CHECK:      ret void
}

// rdar://13113981
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
  // CHECK:      alloca i8*
  // CHECK:      [[CLEANUP_SAVE:%.*]] = alloca i8*
  // CHECK:      [[RUN_CLEANUP:%.*]] = alloca i1
  //   Evaluate condition; cleanup disabled by default.
  // CHECK:      [[T0:%.*]] = load i32, i32* [[COND]],
  // CHECK-NEXT: icmp ne i32 [[T0]], 0
  // CHECK-NEXT: store i1 false, i1* [[RUN_CLEANUP]]
  // CHECK-NEXT: br i1
  //   Within true branch, cleanup enabled.
  // CHECK:      [[T0:%.*]] = call i8* @test2_producer()
  // CHECK-NEXT: [[T1:%.*]] = notail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* [[T0]])
  // CHECK-NEXT: store i8* [[T1]], i8** [[CLEANUP_SAVE]]
  // CHECK-NEXT: store i1 true, i1* [[RUN_CLEANUP]]
  // CHECK-NEXT: br label
  //   Join point for conditional operator; retain immediately.
  // CHECK:      [[T0:%.*]] = phi i8* [ [[T1]], {{%.*}} ], [ null, {{%.*}} ]
  // CHECK-NEXT: [[RESULT:%.*]] = call i8* @llvm.objc.retain(i8* [[T0]])
  //   Leaving full-expression; run conditional cleanup.
  // CHECK-NEXT: [[T0:%.*]] = load i1, i1* [[RUN_CLEANUP]]
  // CHECK-NEXT: br i1 [[T0]]
  // CHECK:      [[T0:%.*]] = load i8*, i8** [[CLEANUP_SAVE]]
  // CHECK-NEXT: call void @llvm.objc.release(i8* [[T0]])
  // CHECK-NEXT: br label
  //   And way down at the end of the loop:
  // CHECK:      call void @llvm.objc.release(i8* [[RESULT]])
}

void test3(int cond) {
  __strong id *p = cond ? (__strong id[]){g0, g1} : (__strong id[]){g1, g0};
  test2(cond);

  // CHECK: define{{.*}} void @test3(
  // CHECK: %[[P:.*]] = alloca i8**, align 8
  // CHECK: %[[_COMPOUNDLITERAL:.*]] = alloca [2 x i8*], align 8
  // CHECK: %[[CLEANUP_COND:.*]] = alloca i1, align 1
  // CHECK: %[[_COMPOUNDLITERAL1:.*]] = alloca [2 x i8*], align 8
  // CHECK: %[[CLEANUP_COND4:.*]] = alloca i1, align 1

  // CHECK: %[[ARRAYINIT_BEGIN:.*]] = getelementptr inbounds [2 x i8*], [2 x i8*]* %[[_COMPOUNDLITERAL]], i64 0, i64 0
  // CHECK: %[[V2:.*]] = load i8*, i8** @g0, align 8
  // CHECK: %[[V3:.*]] = call i8* @llvm.objc.retain(i8* %[[V2]])
  // CHECK: store i8* %[[V3]], i8** %[[ARRAYINIT_BEGIN]], align 8
  // CHECK: %[[ARRAYINIT_ELEMENT:.*]] = getelementptr inbounds i8*, i8** %[[ARRAYINIT_BEGIN]], i64 1
  // CHECK: %[[V4:.*]] = load i8*, i8** @g1, align 8
  // CHECK: %[[V5:.*]] = call i8* @llvm.objc.retain(i8* %[[V4]])
  // CHECK: store i8* %[[V5]], i8** %[[ARRAYINIT_ELEMENT]], align 8
  // CHECK: store i1 true, i1* %[[CLEANUP_COND]], align 1
  // CHECK: %[[ARRAYDECAY:.*]] = getelementptr inbounds [2 x i8*], [2 x i8*]* %[[_COMPOUNDLITERAL]], i64 0, i64 0

  // CHECK: %[[ARRAYINIT_BEGIN2:.*]] = getelementptr inbounds [2 x i8*], [2 x i8*]* %[[_COMPOUNDLITERAL1]], i64 0, i64 0
  // CHECK: %[[V6:.*]] = load i8*, i8** @g1, align 8
  // CHECK: %[[V7:.*]] = call i8* @llvm.objc.retain(i8* %[[V6]])
  // CHECK: store i8* %[[V7]], i8** %[[ARRAYINIT_BEGIN2]], align 8
  // CHECK: %[[ARRAYINIT_ELEMENT3:.*]] = getelementptr inbounds i8*, i8** %[[ARRAYINIT_BEGIN2]], i64 1
  // CHECK: %[[V8:.*]] = load i8*, i8** @g0, align 8
  // CHECK: %[[V9:.*]] = call i8* @llvm.objc.retain(i8* %[[V8]])
  // CHECK: store i8* %[[V9]], i8** %[[ARRAYINIT_ELEMENT3]], align 8
  // CHECK: store i1 true, i1* %[[CLEANUP_COND4]], align 1
  // CHECK: %[[ARRAYDECAY5:.*]] = getelementptr inbounds [2 x i8*], [2 x i8*]* %[[_COMPOUNDLITERAL1]], i64 0, i64 0

  // CHECK: %[[COND6:.*]] = phi i8** [ %[[ARRAYDECAY]], %{{.*}} ], [ %[[ARRAYDECAY5]], %{{.*}} ]
  // CHECK: store i8** %[[COND6]], i8*** %[[P]], align 8
  // CHECK: call void @test2(

  // CHECK: %[[ARRAY_BEGIN:.*]] = getelementptr inbounds [2 x i8*], [2 x i8*]* %[[_COMPOUNDLITERAL1]], i32 0, i32 0
  // CHECK: %[[V11:.*]] = getelementptr inbounds i8*, i8** %[[ARRAY_BEGIN]], i64 2

  // CHECK: %[[ARRAYDESTROY_ELEMENTPAST:.*]] = phi i8** [ %[[V11]], %{{.*}} ], [ %[[ARRAYDESTROY_ELEMENT:.*]], %{{.*}} ]
  // CHECK: %[[ARRAYDESTROY_ELEMENT]] = getelementptr inbounds i8*, i8** %[[ARRAYDESTROY_ELEMENTPAST]], i64 -1
  // CHECK: %[[V12:.*]] = load i8*, i8** %[[ARRAYDESTROY_ELEMENT]], align 8
  // CHECK: call void @llvm.objc.release(i8* %[[V12]])

  // CHECK: %[[ARRAY_BEGIN10:.*]] = getelementptr inbounds [2 x i8*], [2 x i8*]* %[[_COMPOUNDLITERAL]], i32 0, i32 0
  // CHECK: %[[V13:.*]] = getelementptr inbounds i8*, i8** %[[ARRAY_BEGIN10]], i64 2

  // CHECK: %[[ARRAYDESTROY_ELEMENTPAST12:.*]] = phi i8** [ %[[V13]], %{{.*}} ], [ %[[ARRAYDESTROY_ELEMENT13:.*]], %{{.*}} ]
  // CHECK: %[[ARRAYDESTROY_ELEMENT13]] = getelementptr inbounds i8*, i8** %[[ARRAYDESTROY_ELEMENTPAST12]], i64 -1
  // CHECK: %[[V14:.*]] = load i8*, i8** %[[ARRAYDESTROY_ELEMENT13]], align 8
  // CHECK: call void @llvm.objc.release(i8* %[[V14]])
}

// CHECK: attributes [[NUW]] = { nounwind }
