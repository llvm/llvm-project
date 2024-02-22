// RUN: %clang_cc1 -fptrauth-calls -fobjc-arc -fblocks -fobjc-runtime=ios-7 -triple arm64-apple-ios -emit-llvm %s  -o - | FileCheck %s

void (^blockptr)(void);

// CHECK: [[INVOCATION_1:@.*]] =  private constant { ptr, i32, i64, i64 } { ptr {{@.*}}, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ ptr, i32, i32, ptr, ptr }, ptr [[GLOBAL_BLOCK_1:@.*]], i32 0, i32 3) to i64), i64 0 }, section "llvm.ptrauth"
// CHECK: [[GLOBAL_BLOCK_1]] = internal constant { ptr, i32, i32, ptr, ptr } { ptr @_NSConcreteGlobalBlock, i32 1342177280, i32 0, ptr [[INVOCATION_1]],
void (^globalblock)(void) = ^{};

// CHECK: [[COPYDISPOSE_COPY:@.*]] = private constant { ptr, i32, i64, i64 } { ptr {{@.*}}, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ i64, i64, ptr, ptr, ptr, i64 }, ptr [[COPYDISPOSE_DESCRIPTOR:@.*]], i32 0, i32 2) to i64), i64 0 }, section "llvm.ptrauth"
// CHECK: [[COPYDISPOSE_DISPOSE:@.*]] = private constant { ptr, i32, i64, i64 } { ptr {{@.*}}, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ i64, i64, ptr, ptr, ptr, i64 }, ptr [[COPYDISPOSE_DESCRIPTOR]], i32 0, i32 3) to i64), i64 0 }, section "llvm.ptrauth"
// CHECK: [[COPYDISPOSE_DESCRIPTOR:@.*]] = linkonce_odr hidden unnamed_addr constant { i64, i64, ptr, ptr, ptr, i64 } { i64 0, i64 40, ptr [[COPYDISPOSE_COPY]], ptr [[COPYDISPOSE_DISPOSE]],

@interface A
- (int) count;
@end

// CHECK-LABEL: define void @test_block_call()
void test_block_call() {
  // CHECK:      [[BLOCK:%.*]] = load ptr, ptr @blockptr,
  // CHECK-NEXT: [[FNADDR:%.*]] = getelementptr inbounds {{.*}}, ptr [[BLOCK]], i32 0, i32 3
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[FNADDR]],
  // CHECK-NEXT: [[DISC:%.*]] = ptrtoint ptr [[FNADDR]] to i64
  // CHECK-NEXT: call void [[T0]](ptr noundef [[BLOCK]]) [ "ptrauth"(i32 0, i64 [[DISC]]) ]
  blockptr();
}

void use_block(int (^)(void));

// CHECK-LABEL: define void @test_block_literal(
void test_block_literal(int i) {
  // CHECK:      [[I:%.*]] = alloca i32,
  // CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:.*]], align
  // CHECK:      [[FNPTRADDR:%.*]] = getelementptr inbounds [[BLOCK_T]], ptr [[BLOCK]], i32 0, i32 3
  // CHECK-NEXT: [[DISCRIMINATOR:%.*]] = ptrtoint ptr [[FNPTRADDR]] to i64
  // CHECK-NEXT: [[SIGNED:%.*]] = call i64 @llvm.ptrauth.sign(i64 ptrtoint (ptr {{@.*}} to i64), i32 0, i64 [[DISCRIMINATOR]])
  // CHECK-NEXT: [[T0:%.*]] = inttoptr i64 [[SIGNED]] to ptr
  // CHECK-NEXT: store ptr [[T0]], ptr [[FNPTRADDR]]
  use_block(^{return i;});
}

// CHECK-LABEL: define void @test_copy_destroy
void test_copy_destroy(A *a) {
  // CHECK: [[COPYDISPOSE_DESCRIPTOR]]
  use_block(^{return [a count];});
}

// CHECK-LABEL: define void @test_byref_copy_destroy
void test_byref_copy_destroy(A *a) {
  // CHECK:      [[COPY_FIELD:%.*]] = getelementptr inbounds [[BYREF_T:%.*]], ptr [[BYREF:%.*]], i32 0, i32 4
  // CHECK-NEXT: [[T0:%.*]] = ptrtoint ptr [[COPY_FIELD]] to i64
  // CHECK-NEXT: [[T1:%.*]] = call i64 @llvm.ptrauth.sign(i64 ptrtoint (ptr {{@.*}} to i64), i32 0, i64 [[T0]])
  // CHECK-NEXT: [[T2:%.*]] = inttoptr i64 [[T1]] to ptr
  // CHECK-NEXT: store ptr [[T2]], ptr [[COPY_FIELD]], align 8
  // CHECK:      [[DISPOSE_FIELD:%.*]] = getelementptr inbounds [[BYREF_T]], ptr [[BYREF]], i32 0, i32 5
  // CHECK-NEXT: [[T0:%.*]] = ptrtoint ptr [[DISPOSE_FIELD]] to i64
  // CHECK-NEXT: [[T1:%.*]] = call i64 @llvm.ptrauth.sign(i64 ptrtoint (ptr {{@.*}} to i64), i32 0, i64 [[T0]])
  // CHECK-NEXT: [[T2:%.*]] = inttoptr i64 [[T1]] to ptr
  // CHECK-NEXT: store ptr [[T2]], ptr [[DISPOSE_FIELD]], align 8
  __block A *aweak = a;
  use_block(^{return [aweak count];});
}
