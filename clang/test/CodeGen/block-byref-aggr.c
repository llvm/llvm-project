// RUN: %clang_cc1 %s -emit-llvm -o - -fblocks -triple x86_64-apple-darwin10 | FileCheck %s

// CHECK: [[AGG:%.*]] = type { i32 }
typedef struct { int v; } Agg;
Agg makeAgg(void);

// When assigning into a __block variable, ensure that we compute that
// address *after* evaluating the RHS when the RHS has the capacity to
// cause a block copy.
void test0(void) {
  __block Agg a = {100};
  ^{ (void)a; };

 a = makeAgg();
}
// CHECK-LABEL:    define{{.*}} void @test0()
// CHECK:      [[A:%.*]] = alloca [[BYREF:%.*]], align 8
// CHECK-NEXT: alloca <{ ptr, i32, i32, ptr, ptr, ptr }>, align 8
// CHECK-NEXT: [[TEMP:%.*]] = alloca [[AGG]], align 4
// CHECK:      [[RESULT:%.*]] = call i32 @makeAgg()
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[AGG]], ptr [[TEMP]], i32 0, i32 0
// CHECK-NEXT: store i32 [[RESULT]], ptr [[T0]]
//   Check that we properly assign into the forwarding pointer.
// CHECK-NEXT: [[A_FORWARDING:%.*]] = getelementptr inbounds [[BYREF]], ptr [[A]], i32 0, i32 1
// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[A_FORWARDING]]
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[BYREF]], ptr [[T0]], i32 0, i32 4
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[T1]], ptr align 4 [[TEMP]], i64 4, i1 false)
//   Verify that there's nothing else significant in the function.
// CHECK-NEXT: call void @_Block_object_dispose(ptr [[A]], i32 8)
// CHECK-NEXT: ret void

// When chaining assignments into __block variables, make sure we
// propagate the actual value into the outer variable.
void test1(void) {
  __block Agg a, b;
  ^{ (void)a; (void)b; };
  a = b = makeAgg();
}
// CHECK-LABEL:    define{{.*}} void @test1()
// CHECK:      [[A:%.*]] = alloca [[A_BYREF:%.*]], align 8
// CHECK-NEXT: [[B:%.*]] = alloca [[B_BYREF:%.*]], align 8
// CHECK-NEXT: alloca <{ ptr, i32, i32, ptr, ptr, ptr, ptr }>, align 8
// CHECK-NEXT: [[TEMP:%.*]] = alloca [[AGG]], align 4
// CHECK:      [[RESULT:%.*]] = call i32 @makeAgg()
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[AGG]], ptr [[TEMP]], i32 0, i32 0
// CHECK-NEXT: store i32 [[RESULT]], ptr [[T0]]
//   Check that we properly assign into the forwarding pointer, first for b:
// CHECK-NEXT: [[B_FORWARDING:%.*]] = getelementptr inbounds [[B_BYREF]], ptr [[B]], i32 0, i32 1
// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[B_FORWARDING]]
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[B_BYREF]], ptr [[T0]], i32 0, i32 4
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[T1]], ptr align 4 [[TEMP]], i64 4, i1 false)
//   Then for 'a':
// CHECK-NEXT: [[A_FORWARDING:%.*]] = getelementptr inbounds [[A_BYREF]], ptr [[A]], i32 0, i32 1
// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[A_FORWARDING]]
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[A_BYREF]], ptr [[T0]], i32 0, i32 4
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[T1]], ptr align 4 [[TEMP]], i64 4, i1 false)
//   Verify that there's nothing else significant in the function.
// CHECK-NEXT: call void @_Block_object_dispose(ptr [[B]], i32 8)
// CHECK-NEXT: call void @_Block_object_dispose(ptr [[A]], i32 8)
// CHECK-NEXT: ret void
