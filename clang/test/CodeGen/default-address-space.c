// RUN: %clang_cc1 -triple amdgcn -emit-llvm < %s | FileCheck -check-prefixes=PIZ,COM %s
// RUN: %clang_cc1 -triple amdgcn---amdgiz -emit-llvm < %s | FileCheck -check-prefixes=CHECK,COM %s

// PIZ-DAG: @foo = common addrspace(4) global i32 0
// CHECK-DAG: @foo = common global i32 0
int foo;

// PIZ-DAG: @ban = common addrspace(4) global [10 x i32] zeroinitializer
// CHECK-DAG: @ban = common global [10 x i32] zeroinitializer
int ban[10];

// PIZ-DAG: @A = common addrspace(4) global i32 addrspace(4)* null
// PIZ-DAG: @B = common addrspace(4) global i32 addrspace(4)* null
// CHECK-DAG: @A = common global i32* null
// CHECK-DAG: @B = common global i32* null
int *A;
int *B;

// COM-LABEL: define i32 @test1()
// PIZ: load i32, i32 addrspace(4)* @foo
// CHECK: load i32, i32* @foo
int test1() { return foo; }

// COM-LABEL: define i32 @test2(i32 %i)
// PIZ: load i32, i32 addrspace(4)*
// PIZ-NEXT: ret i32
// CHECK: load i32, i32*
// CHECK-NEXT: ret i32
int test2(int i) { return ban[i]; }

// COM-LABEL: define void @test3()
// PIZ: load i32 addrspace(4)*, i32 addrspace(4)* addrspace(4)* @B
// PIZ: load i32, i32 addrspace(4)*
// PIZ: load i32 addrspace(4)*, i32 addrspace(4)* addrspace(4)* @A
// PIZ: store i32 {{.*}}, i32 addrspace(4)*
// CHECK: load i32*, i32** @B
// CHECK: load i32, i32*
// CHECK: load i32*, i32** @A
// CHECK: store i32 {{.*}}, i32*
void test3() {
  *A = *B;
}

// PIZ-LABEL: define void @test4(i32 addrspace(4)* %a)
// PIZ: %[[a_addr:.*]] = alloca i32 addrspace(4)*
// PIZ: store i32 addrspace(4)* %a, i32 addrspace(4)** %[[a_addr]]
// PIZ: %[[r0:.*]] = load i32 addrspace(4)*, i32 addrspace(4)** %[[a_addr]]
// PIZ: %[[arrayidx:.*]] = getelementptr inbounds i32, i32 addrspace(4)* %[[r0]]
// PIZ: store i32 0, i32 addrspace(4)* %[[arrayidx]]
// CHECK-LABEL: define void @test4(i32* %a)
// CHECK: %[[a_addr:.*]] = alloca i32*, align 4, addrspace(5)
// CHECK: store i32* %a, i32* addrspace(5)* %[[a_addr]]
// CHECK: %[[r0:.*]] = load i32*, i32* addrspace(5)* %[[a_addr]]
// CHECK: %[[arrayidx:.*]] = getelementptr inbounds i32, i32* %[[r0]]
// CHECK: store i32 0, i32* %[[arrayidx]]
void test4(int *a) {
  a[0] = 0;
}
