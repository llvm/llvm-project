// RUN: %clang_cc1 -fobjc-arc -fblocks -fptrauth-calls -triple arm64e-apple-ios  -emit-llvm -o - %s | FileCheck %s

_Static_assert(__has_feature(ptrauth_signed_block_descriptors), "-fptrauth-block-descriptor-pointers should set ptrauth_signed_block_descriptors");

void a() {
  // Test out a global block.
  void (^blk)(void) = ^{};
}

// CHECK: [[BLOCK_DESCRIPTOR_NAME:@"__block_descriptor_.*"]] = linkonce_odr hidden unnamed_addr constant { i64, i64, ptr, ptr } { i64 0, i64 32, ptr @.str, ptr null }


// CHECK: @__block_literal_global = internal constant { ptr, i32, i32, ptr, ptr } { ptr @_NSConcreteGlobalBlock, i32 1342177280, i32 0, ptr ptrauth (ptr @__a_block_invoke, i32 0, i64 0, ptr getelementptr inbounds ({ ptr, i32, i32, ptr, ptr }, ptr @__block_literal_global, i32 0, i32 3)), ptr ptrauth (ptr @"__block_descriptor_32_e5_v8\01?0l", i32 2, i64 49339, ptr getelementptr inbounds ({ ptr, i32, i32, ptr, ptr }, ptr @__block_literal_global, i32 0, i32 4)) }

void b(int p) {
  // CHECK-LABEL: define void @b

  // Test out a stack block.
  void (^blk)(void) = ^{(void)p;};

  // CHECK: [[BLOCK:%.*]] = alloca <{ ptr, i32, i32, ptr, ptr, i32 }>
  // CHECK: [[BLOCK_DESCRIPTOR_REF:%.*]] = getelementptr inbounds nuw <{ {{.*}} }>, ptr [[BLOCK]], i32 0, i32 4
  // CHECK: [[BLOCK_DESCRIPTOR_REF_INT:%.*]] = ptrtoint ptr [[BLOCK_DESCRIPTOR_REF]] to i64
  // CHECK: [[BLENDED:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[BLOCK_DESCRIPTOR_REF_INT]], i64 49339)
  // CHECK: [[SIGNED_REF:%.*]] = call i64 @llvm.ptrauth.sign(i64 ptrtoint (ptr @"__block_descriptor_36_e5_v8\01?0l" to i64), i32 2, i64 [[BLENDED]])
  // CHECK: [[SIGNED_REF_PTR:%.*]] = inttoptr i64 [[SIGNED_REF]] to ptr
  // CHECK: store ptr [[SIGNED_REF_PTR]], ptr [[BLOCK_DESCRIPTOR_REF]]
}
