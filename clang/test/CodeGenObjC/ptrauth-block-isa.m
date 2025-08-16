// RUN: %clang_cc1 -fptrauth-calls -fptrauth-objc-isa -fobjc-arc -fblocks -triple arm64e -emit-llvm %s -o - | FileCheck %s

void (^globalblock)(void) = ^{};
// CHECK: @_NSConcreteGlobalBlock.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_NSConcreteGlobalBlock, i32 2, i64 ptrtoint (ptr [[GLOBAL_BLOCK_1:@.*]] to i64), i64 27361 }, section "llvm.ptrauth", align 8
// CHECK: [[INVOCATION_1:@.*]] =  private constant { ptr, i32, i64, i64 } { ptr {{@.*}}, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ ptr, i32, i32, ptr, ptr }, ptr [[GLOBAL_BLOCK_1]], i32 0, i32 3) to i64), i64 0 }, section "llvm.ptrauth"
// CHECK: [[GLOBAL_BLOCK_1]] = internal constant { ptr, i32, i32, ptr, ptr } { ptr @_NSConcreteGlobalBlock.ptrauth, i32 1342177280, i32 0, ptr [[INVOCATION_1]],

@interface A
- (int) count;
@end

void use_block(int (^)(void));

// CHECK-LABEL: define dso_local void @test_block_literal(
void test_block_literal(int i) {
  // CHECK:      [[I:%.*]] = alloca i32,
  // CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:.*]], align
  // CHECK:      [[ISAPTRADDR:%.*]] = getelementptr inbounds nuw [[BLOCK_T]], ptr [[BLOCK]], i32 0, i32 0
  // CHECK-NEXT: [[ISAPTRADDR_I:%.*]] = ptrtoint ptr [[ISAPTRADDR]] to i64
  // CHECK-NEXT: [[ISADISCRIMINATOR:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[ISAPTRADDR_I]], i64 27361)
  // CHECK-NEXT: [[SIGNEDISA:%.*]] = call i64 @llvm.ptrauth.sign(i64 ptrtoint (ptr @_NSConcreteStackBlock to i64), i32 2, i64 [[ISADISCRIMINATOR]])
  // CHECK-NEXT: [[SIGNEDISAPTR:%.*]] = inttoptr i64 [[SIGNEDISA]] to ptr
  // CHECK-NEXT: store ptr [[SIGNEDISAPTR]], ptr [[ISAPTRADDR]]
  use_block(^{return i;});
}

void test_conversion_helper(id);

// CHECK-LABEL: define dso_local void @test_conversion(
void test_conversion(id a) {
  // CHECK:      [[A:%.*addr]] = alloca ptr
  // CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:.*]], align
  // CHECK:      [[ISAPTRADDR:%.*]] = getelementptr inbounds nuw [[BLOCK_T]], ptr [[BLOCK]], i32 0, i32 0
  // CHECK-NEXT: [[ISAPTRADDR_I:%.*]] = ptrtoint ptr [[ISAPTRADDR]] to i64
  // CHECK-NEXT: [[ISADISCRIMINATOR:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[ISAPTRADDR_I]], i64 27361)
  // CHECK-NEXT: [[SIGNEDISA:%.*]] = call i64 @llvm.ptrauth.sign(i64 ptrtoint (ptr @_NSConcreteStackBlock to i64), i32 2, i64 [[ISADISCRIMINATOR]])
  // CHECK-NEXT: [[SIGNEDISAPTR:%.*]] = inttoptr i64 [[SIGNEDISA]] to ptr
  // CHECK-NEXT: store ptr [[SIGNEDISAPTR]], ptr [[ISAPTRADDR]]
  test_conversion_helper(^{
    (void)a;
  });
}
