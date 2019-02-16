// RUN: %clang_cc1 -triple=x86_64-apple-darwin10 -emit-llvm -std=c++11 -fblocks -o - %s | FileCheck %s

// CHECK: %[[BLOCK_CAPTURED0:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i32*, i32* }>, <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i32*, i32* }>* %[[BLOCK:.*]], i32 0, i32 5
// CHECK: %[[V0:.*]] = getelementptr inbounds %[[LAMBDA_CLASS:.*]], %[[LAMBDA_CLASS]]* %[[THIS:.*]], i32 0, i32 0
// CHECK: %[[V1:.*]] = load i32*, i32** %[[V0]], align 8
// CHECK: store i32* %[[V1]], i32** %[[BLOCK_CAPTURED0]], align 8
// CHECK: %[[BLOCK_CAPTURED1:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i32*, i32* }>, <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i32*, i32* }>* %[[BLOCK]], i32 0, i32 6
// CHECK: %[[V2:.*]] = getelementptr inbounds %[[LAMBDA_CLASS]], %[[LAMBDA_CLASS]]* %[[THIS]], i32 0, i32 1
// CHECK: %[[V3:.*]] = load i32*, i32** %[[V2]], align 8
// CHECK: store i32* %[[V3]], i32** %[[BLOCK_CAPTURED1]], align 8

void foo1(int &, int &);

void block_in_lambda(int &s1, int &s2) {
  auto lambda = [&s1, &s2]() {
    auto block = ^{
      foo1(s1, s2);
    };
    block();
  };

  lambda();
}
