// RUN: %clang_cc1 -triple x86_64-apple-darwin10.0.0 -emit-llvm -o - %s -std=c++11 -fblocks -fobjc-arc | FileCheck  %s

// CHECK: define internal void @___Z16test_default_argi_block_invoke(i8* noundef %[[BLOCK_DESCRIPTOR:.*]])
// CHECK: %[[BLOCK:.*]] = bitcast i8* %[[BLOCK_DESCRIPTOR]] to <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i32 }>*
// CHECK: %[[BLOCK_CAPTURE_ADDR:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i32 }>, <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i32 }>* %[[BLOCK]], i32 0, i32 5
// CHECK: %[[V0:.*]] = load i32, i32* %[[BLOCK_CAPTURE_ADDR]]
// CHECK: call void @_Z4foo1i(i32 noundef %[[V0]])

void foo1(int);

void test_default_arg(const int a = 42) {
  auto block = ^{
    foo1(a);
  };
  block();
}
