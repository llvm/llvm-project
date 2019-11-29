// RUN: %clang_cc1 -triple i386-apple-watchos6.0-simulator -emit-llvm -fblocks -fobjc-arc -o - %s | FileCheck %s

// CHECK: %[[STRUCT_S:.*]] = type { i8* }

typedef struct {
  id x;
} S;

// CHECK: define void @test0(i8* %[[A_0:.*]])
// CHECK: %[[A:.*]] = alloca %[[STRUCT_S]], align 4
// CHECK: %[[X:.*]] = getelementptr inbounds %[[STRUCT_S]], %[[STRUCT_S]]* %[[A]], i32 0, i32 0
// CHECK: store i8* %[[A_0]], i8** %[[X]], align 4
// CHECK: %[[V0:.*]] = bitcast %[[STRUCT_S]]* %[[A]] to i8**
// CHECK: call void @__destructor_4_s0(i8** %[[V0]]) #2

void test0(S a) {
}
