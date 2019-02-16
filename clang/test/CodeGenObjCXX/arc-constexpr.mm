// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-arc -o - -std=c++11 %s | FileCheck %s

// CHECK: %[[TYPE:[a-z0-9]+]] = type opaque
// CHECK: @[[CFSTRING:[a-z0-9_]+]] = private global %struct.__NSConstantString_tag

// CHECK: define void @_Z5test1v
// CHECK:   %[[ALLOCA:[A-Z]+]] = alloca %[[TYPE]]*
// CHECK:   %[[V0:[0-9]+]] = call i8* @objc_retain(i8* bitcast (%struct.__NSConstantString_tag* @[[CFSTRING]]
// CHECK:   %[[V1:[0-9]+]] = bitcast i8* %[[V0]] to %[[TYPE]]*
// CHECK:   store %[[TYPE]]* %[[V1]], %[[TYPE]]** %[[ALLOCA]]
// CHECK:   %[[V2:[0-9]+]] = bitcast %[[TYPE]]** %[[ALLOCA]]
// CHECK:   call void @objc_storeStrong(i8** %[[V2]], i8* null)

@class NSString;

void test1() {
  constexpr NSString *S = @"abc";
}
