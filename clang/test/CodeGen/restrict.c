// RUN: %clang_cc1 -triple x86_64-darwin-apple -emit-llvm %s -o - | FileCheck %s

// PR6695

// CHECK: define void @test0(i32* %{{.*}}, i32 %{{.*}})
void test0(int *x, int y) {
}

// CHECK: define void @test1(i32* noalias %{{.*}}, i32 %{{.*}})
void test1(int * restrict x, int y) {
}

// CHECK: define void @test2(i32* %{{.*}}, i32* noalias %{{.*}})
void test2(int *x, int * restrict y) {
}

typedef int * restrict rp;

// CHECK: define void @test3(i32* noalias %{{.*}}, i32 %{{.*}})
void test3(rp x, int y) {
}

// CHECK: define void @test4(i32* %{{.*}}, i32* noalias %{{.*}})
void test4(int *x, rp y) {
}

