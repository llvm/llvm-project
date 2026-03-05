// RUN: %clang_cc1 -triple amdgcn---amdgiz -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple amdgcn---amdgiz -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple amdgcn---amdgiz -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

// LLVM-DAG: @foo = addrspace(1) global i32 0
// LLVM-DAG: @ban = addrspace(1) global [10 x i32] zeroinitializer
// LLVM-DAG: @A = addrspace(1) global ptr null
// LLVM-DAG: @B = addrspace(1) global ptr null
// OGCG-DAG: @foo ={{.*}} addrspace(1) global i32 0
// OGCG-DAG: @ban ={{.*}} addrspace(1) global [10 x i32] zeroinitializer
// OGCG-DAG: @A ={{.*}} addrspace(1) global ptr null
// OGCG-DAG: @B ={{.*}} addrspace(1) global ptr null
int foo;
int ban[10];
int *A;
int *B;

// CIR-LABEL: cir.func {{.*}} @test1
// LLVM-LABEL: define{{.*}} i32 @test1()
// LLVM: alloca i32,{{.*}} addrspace(5)
// LLVM: load i32, ptr addrspace(1) @foo
// OGCG-LABEL: define{{.*}} i32 @test1()
// OGCG: load i32, ptr addrspacecast{{[^@]+}} @foo
int test1(void) { return foo; }

// CIR-LABEL: cir.func {{.*}} @test2
// CIR: %[[I_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i, lang_address_space(offload_private)>, ["i", init]
// CIR: cir.cast address_space %[[I_ALLOCA]] : !cir.ptr<!s32i, lang_address_space(offload_private)> -> !cir.ptr<!s32i>
// LLVM-LABEL: define{{.*}} i32 @test2(i32 %0)
// LLVM: alloca i32,{{.*}} addrspace(5)
// LLVM: addrspacecast ptr addrspace(5)
// LLVM: getelementptr
// LLVM: load i32, ptr
// LLVM: ret i32
// OGCG-LABEL: define{{.*}} i32 @test2(i32 noundef %i)
// OGCG: %[[addr:.*]] = getelementptr
// OGCG: load i32, ptr %[[addr]]
// OGCG-NEXT: ret i32
int test2(int i) { return ban[i]; }

// This is the key test - static alloca with address space cast.
// The alloca is in addrspace(5) and must be cast to generic addrspace(0).
// CIR-LABEL: cir.func {{.*}} @test4
// CIR: %[[A_ALLOCA:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>, lang_address_space(offload_private)>, ["a", init]
// CIR: cir.cast address_space %[[A_ALLOCA]] : !cir.ptr<!cir.ptr<!s32i>, lang_address_space(offload_private)> -> !cir.ptr<!cir.ptr<!s32i>>
// LLVM-LABEL: define{{.*}} void @test4(ptr %0)
// LLVM: %[[alloca:.*]] = alloca ptr,{{.*}} addrspace(5)
// LLVM: %[[a_addr:.*]] = addrspacecast ptr addrspace(5) %[[alloca]] to ptr
// LLVM: store ptr %0, ptr %[[a_addr]]
// LLVM: %[[r0:.*]] = load ptr, ptr %[[a_addr]]
// LLVM: %[[arrayidx:.*]] = getelementptr i32, ptr %[[r0]]
// LLVM: store i32 0, ptr %[[arrayidx]]
// OGCG-LABEL: define{{.*}} void @test4(ptr noundef %a)
// OGCG: %[[alloca:.*]] = alloca ptr, align 8, addrspace(5)
// OGCG: %[[a_addr:.*]] ={{.*}} addrspacecast{{.*}} %[[alloca]] to ptr
// OGCG: store ptr %a, ptr %[[a_addr]]
// OGCG: %[[r0:.*]] = load ptr, ptr %[[a_addr]]
// OGCG: %[[arrayidx:.*]] = getelementptr{{.*}} i32, ptr %[[r0]]
// OGCG: store i32 0, ptr %[[arrayidx]]
void test4(int *a) {
  a[0] = 0;
}

// Test that the return value alloca also gets an address space cast.
// The __retval alloca is in addrspace(5) and stores/loads should go
// through the casted flat pointer.
// CIR-LABEL: cir.func {{.*}} @test_retval
// CIR: %[[RETVAL_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i, lang_address_space(offload_private)>, ["__retval"]
// CIR: %[[RETVAL_CAST:.*]] = cir.cast address_space %[[RETVAL_ALLOCA]] : !cir.ptr<!s32i, lang_address_space(offload_private)> -> !cir.ptr<!s32i>
// CIR: cir.store {{.*}}, %[[RETVAL_CAST]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[RET:.*]] = cir.load {{.*}} %[[RETVAL_CAST]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.return %[[RET]] : !s32i
// LLVM-LABEL: define{{.*}} i32 @test_retval(i32 %{{.*}})
// LLVM-DAG: alloca i32,{{.*}} addrspace(5)
// LLVM-DAG: %[[RETVAL_ALLOCA:.*]] = alloca i32,{{.*}} addrspace(5)
// LLVM-DAG: %[[RETVAL_CAST:.*]] = addrspacecast ptr addrspace(5) %[[RETVAL_ALLOCA]] to ptr
// LLVM: store i32 {{.*}}, ptr %[[RETVAL_CAST]]
// LLVM: %[[RET:.*]] = load i32, ptr %[[RETVAL_CAST]]
// LLVM: ret i32 %[[RET]]
// Note: OGCG optimizes away the store/load through retval for simple returns.
// It stores and loads directly from the parameter, so we only check that
// the retval addrspacecast is generated.
// OGCG-LABEL: define{{.*}} i32 @test_retval(i32 noundef %{{.*}})
// OGCG: %[[RETVAL:.*]] = alloca i32, align 4, addrspace(5)
// OGCG: %[[RETVAL_CAST:.*]] = addrspacecast ptr addrspace(5) %[[RETVAL]] to ptr
// OGCG: ret i32
int test_retval(int x) {
  return x;
}

