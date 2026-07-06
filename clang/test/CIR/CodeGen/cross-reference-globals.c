// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM,LLVMCIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM,OGCG

extern int b1;
int *a1 = &b1;
int b1 = 100;

// CIR-DAG: cir.global external @a1 = #cir.global_view<@b1> : !cir.ptr<!s32i>
// CIR-DAG: cir.global external @b1 = #cir.int<100> : !s32i
// LLVM-DAG: @b1 = global i32 100
// LLVM-DAG: @a1 = global ptr @b1

struct E2 {};
struct B2 { struct E2 e; int x; int y; };
extern struct B2 b2;
int *a2 = &b2.y;
struct B2 b2 = { {}, 10, 20 };
// CIR-DAG: cir.global external @b2 = #cir.const_record<{#cir.int<10> : !s32i, #cir.int<20> : !s32i}> : !rec_B2
// CIR-DAG: cir.global external @a2 = #cir.global_view<@b2, [1 : i32]> : !cir.ptr<!s32i>
// LLVM-DAG: @b2 = global %struct.B2 { i32 10, i32 20 }
// LLVM-DAG: @a2 = global ptr getelementptr {{.*}}(i8, ptr @b2, i64 4)

struct E3 {};
struct In3 { int u, v; };
struct B3 {
  struct E3 e;
  int x;
  struct In3 inner;
  int *self;             // forces 4 bytes alignment padding before self
};
extern struct B3 b3;
int *a3 = &b3.inner.v;
struct B3 b3 = { .x = 7, .self = (int *)&b3 };
// CIR-DAG: cir.global external @b3 = #cir.const_record<{#cir.int<7> : !s32i, #cir.zero : !rec_In3, #cir.global_view<@b3> : !cir.ptr<!s32i>}> : !rec_B3
// CIR-DAG: cir.global external @a3 = #cir.global_view<@b3, [1 : i32, 1 : i32]> : !cir.ptr<!s32i>
// LLVMCIR-DAG: @b3 = global %struct.B3 { i32 7, %struct.In3 zeroinitializer, ptr @b3 }
// OGCG-DAG:    @b3 = global { i32, %struct.In3, [4 x i8], ptr } { i32 7, %struct.In3 zeroinitializer, [4 x i8] zeroinitializer, ptr @b3 }
// LLVM-DAG: @a3 = global ptr getelementptr {{.*}}(i8, ptr @b3, i64 8)

struct E4 {};
struct In4 { int p, q; };
struct B4 {
  struct E4 e;
  int x;
  struct In4 inner;
  int *self;
};
extern struct B4 b4_fwd;
struct A4 { int *target; };
struct A4 a4 = { .target = &b4_fwd.inner.q };
struct B4 b4_fwd = { .x = 11, .self = (int *)&b4_fwd };
// CIR-DAG: cir.global external @b4_fwd = #cir.const_record<{#cir.int<11> : !s32i, #cir.zero : !rec_In4, #cir.global_view<@b4_fwd> : !cir.ptr<!s32i>}> : !rec_B4
// CIR-DAG: cir.global external @a4 = #cir.const_record<{#cir.global_view<@b4_fwd, [1 : i32, 1 : i32]> : !cir.ptr<!s32i>}> : !rec_A4
// LLVMCIR-DAG: @b4_fwd = global %struct.B4 { i32 11, %struct.In4 zeroinitializer, ptr @b4_fwd }
// OGCG-DAG:    @b4_fwd = global { i32, %struct.In4, [4 x i8], ptr } { i32 11, %struct.In4 zeroinitializer, [4 x i8] zeroinitializer, ptr @b4_fwd }
// LLVM-DAG: @a4 = global %struct.A4 { ptr getelementptr {{.*}}(i8, ptr @b4_fwd, i64 8) }

