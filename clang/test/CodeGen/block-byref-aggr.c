// RUN: %clang_cc1 %s -emit-llvm -o - -fblocks -triple x86_64-apple-darwin10 | FileCheck %s

// CHECK: [[AGG:%.*]] = type { i32 }
typedef struct { int v; } Agg;
Agg makeAgg(void);

// When assigning into a __block variable, ensure that we compute that
// address *after* evaluating the RHS when the RHS has the capacity to
// cause a block copy.  rdar://9309454
void test0() {
  __block Agg a = {100};

 a = makeAgg();
}
// CHECK-LABEL:    define void @test0()
// CHECK:      [[A:%.*]] = alloca [[BYREF:%.*]], align 8
// CHECK-NEXT: [[TEMP:%.*]] = alloca [[AGG]], align 4
// CHECK:      [[RESULT:%.*]] = call i32 @makeAgg()
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[TEMP]], i32 0, i32 0
// CHECK-NEXT: store i32 [[RESULT]], i32* [[T0]]
//   Check that we properly assign into the forwarding pointer.
// CHECK-NEXT: [[A_FORWARDING:%.*]] = getelementptr inbounds [[BYREF]], [[BYREF]]* [[A]], i32 0, i32 1
// CHECK-NEXT: [[T0:%.*]] = load [[BYREF]]*, [[BYREF]]** [[A_FORWARDING]]
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[BYREF]], [[BYREF]]* [[T0]], i32 0, i32 4
// CHECK-NEXT: [[T2:%.*]] = bitcast [[AGG]]* [[T1]] to i8*
// CHECK-NEXT: [[T3:%.*]] = bitcast [[AGG]]* [[TEMP]] to i8*
// CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[T2]], i8* align 4 [[T3]], i64 4, i1 false)
//   Verify that there's nothing else significant in the function.
// CHECK-NEXT: [[T0:%.*]] = bitcast [[BYREF]]* [[A]] to i8*
// CHECK-NEXT: call void @_Block_object_dispose(i8* [[T0]], i32 8)
// CHECK-NEXT: ret void

// When chaining assignments into __block variables, make sure we
// propagate the actual value into the outer variable.
// rdar://11757470
void test1() {
  __block Agg a, b;
  a = b = makeAgg();
}
// CHECK-LABEL:    define void @test1()
// CHECK:      [[A:%.*]] = alloca [[A_BYREF:%.*]], align 8
// CHECK-NEXT: [[B:%.*]] = alloca [[B_BYREF:%.*]], align 8
// CHECK-NEXT: [[TEMP:%.*]] = alloca [[AGG]], align 4
// CHECK:      [[RESULT:%.*]] = call i32 @makeAgg()
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[AGG]], [[AGG]]* [[TEMP]], i32 0, i32 0
// CHECK-NEXT: store i32 [[RESULT]], i32* [[T0]]
//   Check that we properly assign into the forwarding pointer, first for b:
// CHECK-NEXT: [[B_FORWARDING:%.*]] = getelementptr inbounds [[B_BYREF]], [[B_BYREF]]* [[B]], i32 0, i32 1
// CHECK-NEXT: [[T0:%.*]] = load [[B_BYREF]]*, [[B_BYREF]]** [[B_FORWARDING]]
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[B_BYREF]], [[B_BYREF]]* [[T0]], i32 0, i32 4
// CHECK-NEXT: [[T2:%.*]] = bitcast [[AGG]]* [[T1]] to i8*
// CHECK-NEXT: [[T3:%.*]] = bitcast [[AGG]]* [[TEMP]] to i8*
// CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[T2]], i8* align 4 [[T3]], i64 4, i1 false)
//   Then for 'a':
// CHECK-NEXT: [[A_FORWARDING:%.*]] = getelementptr inbounds [[A_BYREF]], [[A_BYREF]]* [[A]], i32 0, i32 1
// CHECK-NEXT: [[T0:%.*]] = load [[A_BYREF]]*, [[A_BYREF]]** [[A_FORWARDING]]
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[A_BYREF]], [[A_BYREF]]* [[T0]], i32 0, i32 4
// CHECK-NEXT: [[T2:%.*]] = bitcast [[AGG]]* [[T1]] to i8*
// CHECK-NEXT: [[T3:%.*]] = bitcast [[AGG]]* [[TEMP]] to i8*
// CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[T2]], i8* align 4 [[T3]], i64 4, i1 false)
//   Verify that there's nothing else significant in the function.
// CHECK-NEXT: [[T0:%.*]] = bitcast [[B_BYREF]]* [[B]] to i8*
// CHECK-NEXT: call void @_Block_object_dispose(i8* [[T0]], i32 8)
// CHECK-NEXT: [[T0:%.*]] = bitcast [[A_BYREF]]* [[A]] to i8*
// CHECK-NEXT: call void @_Block_object_dispose(i8* [[T0]], i32 8)
// CHECK-NEXT: ret void
