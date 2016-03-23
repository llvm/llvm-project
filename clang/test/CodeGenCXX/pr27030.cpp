// RUN: %clang_cc1 -emit-llvm -triple=i386-pc-win32 %s -o - | FileCheck %s
struct A {};
struct B : A {};
extern "C" {
extern int B::*a;
void test1() { (int A::*)(a); }
}
// CHECK-LABEL: define void @test1(
// CHECK: %[[load:.*]]       = load i32, i32* @a
// CHECK: %[[memptr_cmp:.*]] = icmp ne i32 %[[load]], -1
// CHECK: br i1 %[[memptr_cmp]]

// CHECK: %[[adj:.*]] = sub nsw i32 %[[load]], 0
// CHECK: %[[nv_adj:.*]] = select i1 true, i32 %[[adj]], i32 0

// CHECK: %[[memptr_converted:.*]] = phi i32 [ -1, {{.*}} ], [ %[[nv_adj]], {{.*}} ]
