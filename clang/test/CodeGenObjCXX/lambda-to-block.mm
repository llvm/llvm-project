// RUN: %clang_cc1 -x objective-c++ -fblocks -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 -std=c++1z -emit-llvm -o - %s | FileCheck %s

// rdar://31385153
// Shouldn't crash!

// CHECK: %[[STRUCT_COPYABLE:.*]] = type { i8 }
// CHECK: %[[STRUCT_BLOCK_DESCRIPTOR:.*]] = type { i64, i64 }
// CHECK: %[[CLASS_ANON:.*]] = type { %[[STRUCT_COPYABLE]] }
// CHECK: %[[CLASS_ANON_0:.*]] = type { %[[STRUCT_COPYABLE]] }
// CHECK: %[[CLASS_ANON_1:.*]] = type { %[[STRUCT_COPYABLE]] }
// CHECK: %[[CLASS_ANON_2:.*]] = type { %[[STRUCT_COPYABLE]] }

// CHECK: @[[BLOCK_DESC0:.*]] = internal constant { i64, i64, i8*, i8*, i8*, i8* } { i64 0, i64 33, i8* bitcast (void (i8*, i8*)* @[[COPY_HELPER0:.*__copy_helper_block_.*]] to i8*), i8* bitcast (void (i8*)* @__destroy_helper_block{{.*}} to i8*), {{.*}}}, align 8
// CHECK: @[[BLOCK_DESC1:.*]] = internal constant { i64, i64, i8*, i8*, i8*, i8* } { i64 0, i64 33, i8* bitcast (void (i8*, i8*)* @[[COPY_HELPER1:.*__copy_helper_block_.*]] to i8*), i8* bitcast (void (i8*)* @__destroy_helper_block{{.*}} to i8*), {{.*}}}, align 8
// CHECK: @[[BLOCK_DESC2:.*]] = internal constant { i64, i64, i8*, i8*, i8*, i8* } { i64 0, i64 33, i8* bitcast (void (i8*, i8*)* @[[COPY_HELPER2:.*__copy_helper_block_.*]] to i8*), i8* bitcast (void (i8*)* @__destroy_helper_block{{.*}} to i8*), {{.*}}}, align 8
// CHECK: @[[BLOCK_DESC3:.*]] = internal constant { i64, i64, i8*, i8*, i8*, i8* } { i64 0, i64 33, i8* bitcast (void (i8*, i8*)* @[[COPY_HELPER3:.*__copy_helper_block_.*]] to i8*), i8* bitcast (void (i8*)* @__destroy_helper_block{{.*}} to i8*), {{.*}}}, align 8

// CHECK: define{{.*}} void @_Z9hasLambda8Copyable(
// CHECK: %[[BLOCK:.*]] = alloca <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, %[[CLASS_ANON]] }>, align 8
// CHECK: %[[BLOCK1:.*]] = alloca <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, %[[CLASS_ANON_0]] }>, align 8
// CHECK: %[[BLOCK_DESCRIPTOR:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, %[[CLASS_ANON]] }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, %[[CLASS_ANON]] }>* %[[BLOCK]], i32 0, i32 4
// CHECK: store %[[STRUCT_BLOCK_DESCRIPTOR]]* bitcast ({ i64, i64, i8*, i8*, i8*, i8* }* @[[BLOCK_DESC0]] to %[[STRUCT_BLOCK_DESCRIPTOR]]*), %[[STRUCT_BLOCK_DESCRIPTOR]]** %[[BLOCK_DESCRIPTOR]], align 8
// CHECK: %[[BLOCK_DESCRIPTOR6:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, %[[CLASS_ANON_0]] }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, %[[CLASS_ANON_0]] }>* %[[BLOCK1]], i32 0, i32 4
// CHECK: store %[[STRUCT_BLOCK_DESCRIPTOR]]* bitcast ({ i64, i64, i8*, i8*, i8*, i8* }* @[[BLOCK_DESC1]] to %[[STRUCT_BLOCK_DESCRIPTOR]]*), %[[STRUCT_BLOCK_DESCRIPTOR]]** %[[BLOCK_DESCRIPTOR6]], align 8

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
// CHECK: %[[CALL:.*]] = call void ()* @[[CONV_FUNC0:.*]](%[[CLASS_ANON_1]]*
// CHECK: call void @_Z10takesBlockU13block_pointerFvvE(void ()* %[[CALL]])
// CHECK: %[[CALL1:.*]] = call void ()* @[[CONV_FUNC0]](%[[CLASS_ANON_1]]*
// CHECK: call void @_Z10takesBlockU13block_pointerFvvE(void ()* %[[CALL1]])
// CHECK: %[[CALL2:.*]] = call void ()* @[[CONV_FUNC1:.*]](%[[CLASS_ANON_2]]*
// CHECK: call void @_Z10takesBlockU13block_pointerFvvE(void ()* %[[CALL2]])

// CHECK: define internal void ()* @[[CONV_FUNC0]](
// CHECK: %[[BLOCK_DESCRIPTOR:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, %[[CLASS_ANON_1]] }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, %[[CLASS_ANON_1]] }>* %{{.*}}, i32 0, i32 4
// CHECK: store %[[STRUCT_BLOCK_DESCRIPTOR]]* bitcast ({ i64, i64, i8*, i8*, i8*, i8* }* @[[BLOCK_DESC2]] to %[[STRUCT_BLOCK_DESCRIPTOR]]*), %[[STRUCT_BLOCK_DESCRIPTOR]]** %[[BLOCK_DESCRIPTOR]], align 8

// CHECK: define internal void ()* @[[CONV_FUNC1]](
// CHECK: %[[BLOCK_DESCRIPTOR:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, %[[CLASS_ANON_2]] }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, %[[CLASS_ANON_2]] }>* %{{.*}}, i32 0, i32 4
// CHECK: store %[[STRUCT_BLOCK_DESCRIPTOR]]* bitcast ({ i64, i64, i8*, i8*, i8*, i8* }* @[[BLOCK_DESC3]] to %[[STRUCT_BLOCK_DESCRIPTOR]]*), %[[STRUCT_BLOCK_DESCRIPTOR]]** %[[BLOCK_DESCRIPTOR]], align 8

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
