// RUN: %clang_cc1 -triple amdgcn---amdgiz -emit-llvm < %s | FileCheck -check-prefixes=CHECK %s

// CHECK-DAG: @foo ={{.*}} addrspace(1) global i32 0
int foo;

// CHECK-DAG: @ban ={{.*}} addrspace(1) global [10 x i32] zeroinitializer
int ban[10];

// CHECK-DAG: @A ={{.*}} addrspace(1) global ptr null
// CHECK-DAG: @B ={{.*}} addrspace(1) global ptr null
int *A;
int *B;

// CHECK-LABEL: define{{.*}} i32 @test1()
// CHECK: load i32, ptr addrspacecast{{[^@]+}} @foo
int test1(void) { return foo; }

// CHECK-LABEL: define{{.*}} i32 @test2(i32 noundef %i)
// CHECK: %[[addr:.*]] = getelementptr
// CHECK: load i32, ptr %[[addr]]
// CHECK-NEXT: ret i32
int test2(int i) { return ban[i]; }

// CHECK-LABEL: define{{.*}} void @test3()
// CHECK: load ptr, ptr addrspacecast{{.*}} @B
// CHECK: load i32, ptr
// CHECK: load ptr, ptr addrspacecast{{.*}} @A
// CHECK: store i32 {{.*}}, ptr
void test3(void) {
  *A = *B;
}

// CHECK-LABEL: define{{.*}} void @test4(ptr noundef %a)
// CHECK: %[[alloca:.*]] = alloca ptr, align 8, addrspace(5)
// CHECK: %[[a_addr:.*]] ={{.*}} addrspacecast{{.*}} %[[alloca]] to ptr
// CHECK: store ptr %a, ptr %[[a_addr]]
// CHECK: %[[r0:.*]] = load ptr, ptr %[[a_addr]]
// CHECK: %[[arrayidx:.*]] = getelementptr inbounds i32, ptr %[[r0]]
// CHECK: store i32 0, ptr %[[arrayidx]]
void test4(int *a) {
  a[0] = 0;
}
