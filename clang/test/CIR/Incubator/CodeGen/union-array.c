// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -fno-clangir-call-conv-lowering %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -fno-clangir-call-conv-lowering %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

typedef struct {
  char a;
} S_1;

typedef struct {
  long a, b;
} S_2;

typedef union {
  S_1 a;
  S_2 b;
} U;

typedef union {
   int f0;
   int f1;
} U1;

static U1 g = {5};
// LLVM: @__const.bar.x = private constant [2 x ptr] [ptr @g, ptr @g]
// LLVM: @g = internal global { i32 } { i32 5 }
// FIXME: LLVM output should be: @g = internal global %union.U { i32 5 }

// LLVM: @g2 = global ptr getelementptr inbounds nuw (i8, ptr @g1, i64 24)

void foo() { U arr[2] = {{.b = {1, 2}}, {.a = {1}}}; }

// CIR: cir.const #cir.const_record<{#cir.const_record<{#cir.const_record<{#cir.int<1> : !s64i, #cir.int<2> : !s64i}> : {{.*}}}> : {{.*}}, #cir.const_record<{#cir.const_record<{#cir.int<1> : !s8i}> : {{.*}}, #cir.const_array<[#cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i]> : !cir.array<!u8i x 15>}>
// LLVM: store { { %struct.S_2 }, { %struct.S_1, [15 x i8] } } { { %struct.S_2 } { %struct.S_2 { i64 1, i64 2 } }, { %struct.S_1, [15 x i8] } { %struct.S_1 { i8 1 }, [15 x i8] zeroinitializer } }

void bar(void) {
  int *x[2] = { &g.f0, &g.f0 };
}
// CIR: cir.global "private" internal dso_local @g = #cir.const_record<{#cir.int<5> : !s32i}> : !rec_anon_struct
// CIR: cir.const #cir.const_array<[#cir.global_view<@g> : !cir.ptr<!s32i>, #cir.global_view<@g> : !cir.ptr<!s32i>]> : !cir.array<!cir.ptr<!s32i> x 2>

typedef struct {
    long s0;
    int  s1;
} S_3;

typedef union {
   int  f0;
   S_3 f1;
} U2;


static U2 g1[3] = {{0x42},{0x42},{0x42}};
int* g2 = &g1[1].f1.s1;
// CIR: cir.global external @g2 = #cir.global_view<@g1, [1, 1, 4]> : !cir.ptr<!s32i>

void baz(void) {
  (*g2) = 4;
}
