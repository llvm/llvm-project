// RUN: %clang_cc1 -std=c23 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c23 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM

struct S { int a; int b; };

// CIR-DAG: cir.global "private" internal @".compoundliteral" = #cir.int<0> : !s32i align(4)
// CIR-DAG: cir.global "private" internal tls_model = tls_dyn @".compoundliteral.1" = #cir.int<0> : !s32i align(4)
// CIR-DAG: cir.global "private" constant internal @".compoundliteral.2" = #cir.int<3> : !s32i align(4)
// CIR-DAG: cir.global "private" internal @".compoundliteral.3" = #cir.const_record<{#cir.int<4> : !s32i, #cir.int<5> : !s32i}> : !rec_S align(4)
// CIR-DAG: cir.global "private" internal tls_model = tls_dyn @".compoundliteral.4" = #cir.const_record<{#cir.int<6> : !s32i, #cir.int<7> : !s32i}> : !rec_S align(4)
// CIR-DAG: cir.global "private" internal @".compoundliteral.5" = #cir.ptr<null> : !cir.ptr<!s32i> align(8)

// LLVM-DAG: @.compoundliteral = internal global i32 0, align 4
// LLVM-DAG: @.compoundliteral.1 = internal thread_local global i32 0, align 4
// LLVM-DAG: @.compoundliteral.2 = internal constant i32 3, align 4
// LLVM-DAG: @.compoundliteral.3 = internal global %struct.S { i32 4, i32 5 }, align 4
// LLVM-DAG: @.compoundliteral.4 = internal thread_local global %struct.S { i32 6, i32 7 }, align 4
// LLVM-DAG: @.compoundliteral.5 = internal global ptr null, align 8

// CIR-LABEL: @test1()
// CIR: %[[ADDR:.*]] = cir.get_global @".compoundliteral" : !cir.ptr<!s32i>
// CIR-NEXT: %[[VALUE:.*]] = cir.load align(4) %[[ADDR]]
// CIR-NEXT: %[[INC:.*]] = cir.inc nsw %[[VALUE]]
// CIR-NEXT: cir.store align(4) %[[INC]], %[[ADDR]]
// LLVM-LABEL: @test1()
// LLVM: %[[VALUE:.*]] = load i32, ptr @.compoundliteral, align 4
// LLVM-NEXT: %[[INC:.*]] = add nsw i32 %[[VALUE]], 1
// LLVM-NEXT: store i32 %[[INC]], ptr @.compoundliteral, align 4
int test1(void) {
  return ++(static int){0};
}

// CIR-LABEL: @test2()
// CIR: %[[ADDR:.*]] = cir.get_global thread_local @".compoundliteral.1" : !cir.ptr<!s32i>
// CIR-NEXT: %[[VALUE:.*]] = cir.load align(4) %[[ADDR]]
// CIR-NEXT: %[[INC:.*]] = cir.inc nsw %[[VALUE]]
// CIR-NEXT: cir.store align(4) %[[INC]], %[[ADDR]]
// LLVM-LABEL: @test2()
// LLVM: %[[ADDR:.*]] = call{{.*}} ptr @llvm.threadlocal.address.p0(ptr{{.*}} @.compoundliteral.1)
// LLVM-NEXT: %[[VALUE:.*]] = load i32, ptr %[[ADDR]], align 4
// LLVM-NEXT: %[[INC:.*]] = add nsw i32 %[[VALUE]], 1
// LLVM-NEXT: store i32 %[[INC]], ptr %[[ADDR]], align 4
int test2(void) {
  return ++(static thread_local int){0};
}

// CIR-LABEL: @test3()
// CIR: cir.get_global @".compoundliteral.2" : !cir.ptr<!s32i>
// LLVM-LABEL: @test3()
// LLVM: store ptr @.compoundliteral.2, ptr
const int *test3(void) {
  return &(static constexpr int){3};
}

// CIR-LABEL: @test4()
// CIR: %[[ADDR:.*]] = cir.get_global @".compoundliteral.3" : !cir.ptr<!rec_S>
// CIR: cir.copy %[[ADDR]] align(4) to
// LLVM-LABEL: @test4()
// LLVM: call void @llvm.memcpy{{.*}}(ptr{{.*}}, ptr align 4 @.compoundliteral.3, i64 8, i1 false)
struct S test4(void) {
  return (static struct S){4, 5};
}

// CIR-LABEL: @test5()
// CIR: %[[ADDR:.*]] = cir.get_global thread_local @".compoundliteral.4" : !cir.ptr<!rec_S>
// CIR: cir.copy %[[ADDR]] align(4) to
// LLVM-LABEL: @test5()
// LLVM: %[[ADDR:.*]] = call{{.*}} ptr @llvm.threadlocal.address.p0(ptr{{.*}} @.compoundliteral.4)
// LLVM: call void @llvm.memcpy{{.*}}(ptr{{.*}}, ptr align 4 %[[ADDR]], i64 8, i1 false)
struct S test5(void) {
  return (static thread_local struct S){6, 7};
}

// CIR-LABEL: @test6(
// CIR: cir.inc nsw
// CIR: cir.get_global @".compoundliteral.5"
// LLVM-LABEL: @test6(
// LLVM: add nsw i32 {{.*}}, 1
// LLVM: load ptr, ptr @.compoundliteral.5, align 8
int test6(int a) {
  (void)(static int (*)[a++]){0};
  return a;
}
