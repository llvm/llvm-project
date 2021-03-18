// RUN: %clang_cc1 -fptrauth-calls -fobjc-arc -fblocks -fobjc-runtime=ios-7 -triple arm64-apple-ios -emit-llvm %s  -o - | FileCheck %s

void (^blockptr)(void);

// CHECK: [[INVOCATION_1:@.*]] =  private constant { i8*, i32, i64, i64 } { i8* bitcast (void (i8*)* {{@.*}} to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ i8**, i32, i32, i8*, %struct.__block_descriptor* }, { i8**, i32, i32, i8*, %struct.__block_descriptor* }* [[GLOBAL_BLOCK_1:@.*]], i32 0, i32 3) to i64), i64 0 }, section "llvm.ptrauth"
// CHECK: [[GLOBAL_BLOCK_1]] = internal constant { i8**, i32, i32, i8*, %struct.__block_descriptor* } { i8** @_NSConcreteGlobalBlock, i32 1342177280, i32 0, i8* bitcast ({ i8*, i32, i64, i64 }* [[INVOCATION_1]] to i8*),
void (^globalblock)(void) = ^{};

// CHECK: [[COPYDISPOSE_COPY:@.*]] = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (i8*, i8*)* {{@.*}} to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ i64, i64, i8*, i8*, i8*, i64 }, { i64, i64, i8*, i8*, i8*, i64 }* [[COPYDISPOSE_DESCRIPTOR:@.*]], i32 0, i32 2) to i64), i64 0 }, section "llvm.ptrauth"
// CHECK: [[COPYDISPOSE_DISPOSE:@.*]] = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (i8*)* {{@.*}} to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ i64, i64, i8*, i8*, i8*, i64 }, { i64, i64, i8*, i8*, i8*, i64 }* [[COPYDISPOSE_DESCRIPTOR]], i32 0, i32 3) to i64), i64 0 }, section "llvm.ptrauth"
// CHECK: [[COPYDISPOSE_DESCRIPTOR:@.*]] = linkonce_odr hidden unnamed_addr constant { i64, i64, i8*, i8*, i8*, i64 } { i64 0, i64 40, i8* bitcast ({ i8*, i32, i64, i64 }* [[COPYDISPOSE_COPY]] to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* [[COPYDISPOSE_DISPOSE]] to i8*),

@interface A
- (int) count;
@end

// CHECK-LABEL: define void @test_block_call()
void test_block_call() {
  // CHECK:      [[T0:%.*]] = load void ()*, void ()** @blockptr,
  // CHECK-NEXT: [[BLOCK:%.*]] = bitcast void ()* [[T0]] to [[BLOCK_T:%.*]]*{{$}}
  // CHECK-NEXT: [[FNADDR:%.*]] = getelementptr inbounds [[BLOCK_T]], [[BLOCK_T]]* [[BLOCK]], i32 0, i32 3
  // CHECK-NEXT: [[BLOCK_OPAQUE:%.*]] = bitcast [[BLOCK_T]]* [[BLOCK]] to i8*
  // CHECK-NEXT: [[T0:%.*]] = load i8*, i8** [[FNADDR]],
  // CHECK-NEXT: [[FNPTR:%.*]] = bitcast i8* [[T0]] to void (i8*)*
  // CHECK-NEXT: [[DISC:%.*]] = ptrtoint i8** [[FNADDR]] to i64
  // CHECK-NEXT: call void [[FNPTR]](i8* [[BLOCK_OPAQUE]]) [ "ptrauth"(i32 0, i64 [[DISC]]) ]
  blockptr();
}

void use_block(int (^)(void));

// CHECK-LABEL: define void @test_block_literal(
void test_block_literal(int i) {
  // CHECK:      [[I:%.*]] = alloca i32,
  // CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:.*]], align
  // CHECK:      [[FNPTRADDR:%.*]] = getelementptr inbounds [[BLOCK_T]], [[BLOCK_T]]* [[BLOCK]], i32 0, i32 3
  // CHECK-NEXT: [[DISCRIMINATOR:%.*]] = ptrtoint i8** [[FNPTRADDR]] to i64
  // CHECK-NEXT: [[SIGNED:%.*]] = call i64 @llvm.ptrauth.sign.i64(i64 ptrtoint (i32 (i8*)* {{@.*}} to i64), i32 0, i64 [[DISCRIMINATOR]])
  // CHECK-NEXT: [[T0:%.*]] = inttoptr i64 [[SIGNED]] to i8*
  // CHECK-NEXT: store i8* [[T0]], i8** [[FNPTRADDR]]
  use_block(^{return i;});
}

// CHECK-LABEL: define void @test_copy_destroy
void test_copy_destroy(A *a) {
  // CHECK: [[COPYDISPOSE_DESCRIPTOR]]
  use_block(^{return [a count];});
}

// CHECK-LABEL: define void @test_byref_copy_destroy
void test_byref_copy_destroy(A *a) {
  // CHECK:      [[COPY_FIELD:%.*]] = getelementptr inbounds [[BYREF_T:%.*]], {{%.*}}* [[BYREF:%.*]], i32 0, i32 4
  // CHECK-NEXT: [[T0:%.*]] = ptrtoint i8** [[COPY_FIELD]] to i64
  // CHECK-NEXT: [[T1:%.*]] = call i64 @llvm.ptrauth.sign.i64(i64 ptrtoint (void (i8*, i8*)* {{@.*}} to i64), i32 0, i64 [[T0]])
  // CHECK-NEXT: [[T2:%.*]] = inttoptr i64 [[T1]] to i8*
  // CHECK-NEXT: store i8* [[T2]], i8** [[COPY_FIELD]], align 8
  // CHECK:      [[DISPOSE_FIELD:%.*]] = getelementptr inbounds [[BYREF_T]], [[BYREF_T]]* [[BYREF]], i32 0, i32 5
  // CHECK-NEXT: [[T0:%.*]] = ptrtoint i8** [[DISPOSE_FIELD]] to i64
  // CHECK-NEXT: [[T1:%.*]] = call i64 @llvm.ptrauth.sign.i64(i64 ptrtoint (void (i8*)* {{@.*}} to i64), i32 0, i64 [[T0]])
  // CHECK-NEXT: [[T2:%.*]] = inttoptr i64 [[T1]] to i8*
  // CHECK-NEXT: store i8* [[T2]], i8** [[DISPOSE_FIELD]], align 8
  __block A *aweak = a;
  use_block(^{return [aweak count];});
}
