// RUN: %clang_cc1 -triple x86_64-darwin-apple -emit-llvm %s -o - | FileCheck %s

// PR6695

// CHECK: define{{.*}} void @test0(i32* noundef %{{.*}}, i32 noundef %{{.*}})
void test0(int *x, int y) {
}

// CHECK: define{{.*}} void @test1(i32* noalias noundef %{{.*}}, i32 noundef %{{.*}})
void test1(int * restrict x, int y) {
}

// CHECK: define{{.*}} void @test2(i32* noundef %{{.*}}, i32* noalias noundef %{{.*}})
void test2(int *x, int * restrict y) {
}

typedef int * restrict rp;

// CHECK: define{{.*}} void @test3(i32* noalias noundef %{{.*}}, i32 noundef %{{.*}})
void test3(rp x, int y) {
}

// CHECK: define{{.*}} void @test4(i32* noundef %{{.*}}, i32* noalias noundef %{{.*}})
void test4(int *x, rp y) {
}

