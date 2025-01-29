// RUN: %clang_cc1 -x objective-c++ -fblocks -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 -std=c++1z -emit-llvm -o - %s | FileCheck %s

// Shouldn't crash!

// CHECK: %[[CLASS_ANON:.*]] = type { i8 }
// CHECK: %[[CLASS_ANON_0:.*]] = type { i8 }
// CHECK: %[[CLASS_ANON_1:.*]] = type { i8 }
// CHECK: %[[CLASS_ANON_2:.*]] = type { i8 }

// CHECK: @[[BLOCK_DESC0:.*]] = internal constant { i64, i64, ptr, ptr, ptr, ptr } { i64 0, i64 33, ptr @[[COPY_HELPER0:.*__copy_helper_block_.*]], ptr @__destroy_helper_block{{.*}}, {{.*}}}, align 8
// CHECK: @[[BLOCK_DESC1:.*]] = internal constant { i64, i64, ptr, ptr, ptr, ptr } { i64 0, i64 33, ptr @[[COPY_HELPER1:.*__copy_helper_block_.*]], ptr @__destroy_helper_block{{.*}}, {{.*}}}, align 8
// CHECK: @[[BLOCK_DESC2:.*]] = internal constant { i64, i64, ptr, ptr, ptr, ptr } { i64 0, i64 33, ptr @[[COPY_HELPER2:.*__copy_helper_block_.*]], ptr @__destroy_helper_block{{.*}}, {{.*}}}, align 8
// CHECK: @[[BLOCK_DESC3:.*]] = internal constant { i64, i64, ptr, ptr, ptr, ptr } { i64 0, i64 33, ptr @[[COPY_HELPER3:.*__copy_helper_block_.*]], ptr @__destroy_helper_block{{.*}}, {{.*}}}, align 8

// CHECK: define{{.*}} void @_Z9hasLambda8Copyable(
// CHECK: %[[BLOCK:.*]] = alloca <{ ptr, i32, i32, ptr, ptr, %[[CLASS_ANON]] }>, align 8
// CHECK: %[[BLOCK1:.*]] = alloca <{ ptr, i32, i32, ptr, ptr, %[[CLASS_ANON_0]] }>, align 8
// CHECK: %[[BLOCK_DESCRIPTOR:.*]] = getelementptr inbounds nuw <{ ptr, i32, i32, ptr, ptr, %[[CLASS_ANON]] }>, ptr %[[BLOCK]], i32 0, i32 4
// CHECK: store ptr @[[BLOCK_DESC0]], ptr %[[BLOCK_DESCRIPTOR]], align 8
// CHECK: %[[BLOCK_DESCRIPTOR6:.*]] = getelementptr inbounds nuw <{ ptr, i32, i32, ptr, ptr, %[[CLASS_ANON_0]] }>, ptr %[[BLOCK1]], i32 0, i32 4
// CHECK: store ptr @[[BLOCK_DESC1]], ptr %[[BLOCK_DESCRIPTOR6]], align 8

void takesBlock(void (^)(void));

struct Copyable {
  Copyable(const Copyable &x);
};

// Check that each block has its block descriptor and helper function.

void hasLambda(Copyable x) {
  takesBlock([x] () { });
  takesBlock([x] () { });
}
// CHECK: define internal void @[[COPY_HELPER0]]
// CHECK: call void @"_ZZ9hasLambda8CopyableEN3$_0C1ERKS0_"
// CHECK: define internal void @[[COPY_HELPER1]]

// CHECK: define{{.*}} void @_Z17testHelperMerging8Copyable(
// CHECK: %[[CALL:.*]] = call noundef ptr @[[CONV_FUNC0:.*]](ptr
// CHECK: call void @_Z10takesBlockU13block_pointerFvvE(ptr noundef %[[CALL]])
// CHECK: %[[CALL1:.*]] = call noundef ptr @[[CONV_FUNC0]](ptr
// CHECK: call void @_Z10takesBlockU13block_pointerFvvE(ptr noundef %[[CALL1]])
// CHECK: %[[CALL2:.*]] = call noundef ptr @[[CONV_FUNC1:.*]](ptr
// CHECK: call void @_Z10takesBlockU13block_pointerFvvE(ptr noundef %[[CALL2]])

// CHECK: define internal noundef ptr @[[CONV_FUNC0]](
// CHECK: %[[BLOCK_DESCRIPTOR:.*]] = getelementptr inbounds nuw <{ ptr, i32, i32, ptr, ptr, %[[CLASS_ANON_1]] }>, ptr %{{.*}}, i32 0, i32 4
// CHECK: store ptr @[[BLOCK_DESC2]], ptr %[[BLOCK_DESCRIPTOR]], align 8

// CHECK: define internal noundef ptr @[[CONV_FUNC1]](
// CHECK: %[[BLOCK_DESCRIPTOR:.*]] = getelementptr inbounds nuw <{ ptr, i32, i32, ptr, ptr, %[[CLASS_ANON_2]] }>, ptr %{{.*}}, i32 0, i32 4
// CHECK: store ptr @[[BLOCK_DESC3]], ptr %[[BLOCK_DESCRIPTOR]], align 8

// CHECK-LABEL: define internal void @"_ZZ9hasLambda8CopyableEN3$_0C2ERKS0_"
// CHECK: call void @_ZN8CopyableC1ERKS_

// CHECK: define internal void @[[COPY_HELPER2]]
// CHECK: define internal void @[[COPY_HELPER3]]

void testHelperMerging(Copyable x) {
  auto lambda0 = [x]{};
  auto lambda1 = [x]{};
  takesBlock(lambda0);

  // This block has the same helper functions and a descriptor as the block
  // created above.
  takesBlock(lambda0);

  // This block has different helper functions and a descriptor as the blocks
  // created above.
  takesBlock(lambda1);
}
