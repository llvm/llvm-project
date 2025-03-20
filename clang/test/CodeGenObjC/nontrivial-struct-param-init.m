// RUN: %clang_cc1 -triple i386-apple-watchos6.0-simulator -emit-llvm -fblocks -fobjc-arc -o - %s | FileCheck %s

// CHECK: %[[STRUCT_S:.*]] = type { ptr }

typedef struct {
  id x;
} S;

// CHECK: define{{.*}} void @test0(ptr %[[A_0:.*]])
// CHECK: %[[A:.*]] = alloca %[[STRUCT_S]], align 4
// CHECK: %[[X:.*]] = getelementptr inbounds nuw %[[STRUCT_S]], ptr %[[A]], i32 0, i32 0
// CHECK: store ptr %[[A_0]], ptr %[[X]], align 4
// CHECK: call void @__destructor_4_s0(ptr %[[A]]) #2

void test0(S a) {
}
