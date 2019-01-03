// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -O2 -disable-llvm-passes -o - %s | FileCheck %s

void test0(_Bool cond) {
  id test0_helper(void) __attribute__((ns_returns_retained));

  // CHECK-LABEL:      define void @test0(
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

  // CHECK-LABEL:    define void @test1(
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

  // CHECK-LABEL:    define void @test2(
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
  // CHECK-NEXT: [[T1:%.*]] = call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* [[T0]])
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

// CHECK: attributes [[NUW]] = { nounwind }
