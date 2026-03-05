// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

typedef struct {
   int f0 : 24;
   int f1;
   int f2;
} S;

static S g1 = {2799, 9, 123};
static int *g2[4] = {&g1.f1, &g1.f1, &g1.f1, &g1.f1};
static int **g3 = &g2[1];
static int ***g4 = &g3;
static int ****g5 = &g4;

static S g6[2] = {{2799, 9, 123}, {2799, 9, 123}};
static int *g7[2] = {&g6[0].f2, &g6[1].f2};
static int **g8 = &g7[1];

// CHECK-DAG: !rec_anon_struct = !cir.record<struct  {!u8i, !u8i, !u8i, !u8i, !s32i, !s32i}>
// CHECK-DAG: !rec_anon_struct1 = !cir.record<struct  {!s8i, !cir.array<!u8i x 3>, !s32i}>
// CHECK-DAG: !rec_anon_struct2 = !cir.record<struct  {!u8i, !u8i, !u8i, !u8i, !u8i, !u8i, !u8i, !u8i, !rec_S4}>
// CHECK-DAG: !rec_anon_struct3 = !cir.record<struct  {!s16i, !cir.array<!u8i x 2>, !s32i, !s8i, !cir.array<!u8i x 3>}>

// CHECK-DAG: g1 = #cir.const_record<{#cir.int<239> : !u8i, #cir.int<10> : !u8i, #cir.int<0> : !u8i, #cir.zero : !u8i, #cir.int<9> : !s32i, #cir.int<123> : !s32i}> : !rec_anon_struct
// CHECK-DAG: g2 = #cir.const_array<[#cir.global_view<@g1, [4]> : !cir.ptr<!rec_anon_struct>, #cir.global_view<@g1, [4]> : !cir.ptr<!rec_anon_struct>, #cir.global_view<@g1, [4]> : !cir.ptr<!rec_anon_struct>, #cir.global_view<@g1, [4]> : !cir.ptr<!rec_anon_struct>]> : !cir.array<!cir.ptr<!s32i> x 4>
// CHECK-DAG: g3 = #cir.global_view<@g2, [1 : i32]> : !cir.ptr<!cir.ptr<!s32i>>
// CHECK-DAG: g4 = #cir.global_view<@g3> : !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>
// CHECK-DAG: g5 = #cir.global_view<@g4> : !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>>
// CHECK-DAG: g6 = #cir.const_array<[#cir.const_record<{#cir.int<239> : !u8i, #cir.int<10> : !u8i, #cir.int<0> : !u8i, #cir.zero : !u8i, #cir.int<9> : !s32i, #cir.int<123> : !s32i}> : !rec_anon_struct, #cir.const_record<{#cir.int<239> : !u8i, #cir.int<10> : !u8i, #cir.int<0> : !u8i, #cir.zero : !u8i, #cir.int<9> : !s32i, #cir.int<123> : !s32i}> : !rec_anon_struct]> : !cir.array<!rec_anon_struct x 2> 
// CHECK-DAG: g7 = #cir.const_array<[#cir.global_view<@g6, [0, 5]> : !cir.ptr<!s32i>, #cir.global_view<@g6, [1, 5]> : !cir.ptr<!s32i>]> : !cir.array<!cir.ptr<!s32i> x 2> 
// CHECK-DAG: g8 = #cir.global_view<@g7, [1 : i32]> : !cir.ptr<!cir.ptr<!s32i>> 

// LLVM-DAG: @g1 = internal global { i8, i8, i8, i8, i32, i32 } { i8 -17, i8 10, i8 0, i8 0, i32 9, i32 123 }, align 4
// LLVM-DAG: @g2 = internal global [4 x ptr] [ptr getelementptr inbounds nuw (i8, ptr @g1, i64 4), ptr getelementptr inbounds nuw (i8, ptr @g1, i64 4), ptr getelementptr inbounds nuw (i8, ptr @g1, i64 4), ptr getelementptr inbounds nuw (i8, ptr @g1, i64 4)], align 16
// LLVM-DAG: @g3 = internal global ptr getelementptr inbounds nuw (i8, ptr @g2, i64 8), align 8
// LLVM-DAG: @g4 = internal global ptr @g3, align 8
// LLVM-DAG: @g5 = internal global ptr @g4, align 8
// LLVM-DAG: @g6 = internal global [2 x { i8, i8, i8, i8, i32, i32 }] [{ i8, i8, i8, i8, i32, i32 } { i8 -17, i8 10, i8 0, i8 0, i32 9, i32 123 }, { i8, i8, i8, i8, i32, i32 } { i8 -17, i8 10, i8 0, i8 0, i32 9, i32 123 }], align 16
// LLVM-DAG: @g7 = internal global [2 x ptr] [ptr getelementptr inbounds nuw (i8, ptr @g6, i64 8), ptr getelementptr inbounds nuw (i8, ptr @g6, i64 20)], align 16
// LLVM-DAG: @g8 = internal global ptr getelementptr inbounds nuw (i8, ptr @g7, i64 8), align 8

typedef struct {
   char f1;
   int  f6;
} S1;

S1 g9 = {1, 42};
int* g10 = &g9.f6;

#pragma pack(push)
#pragma pack(1)
typedef struct {
   char f1;
   int  f6;
} S2;
#pragma pack(pop)

S2 g11 = {1, 42};
int* g12 = &g11.f6;

// CHECK-DAG: g9 = #cir.const_record<{#cir.int<1> : !s8i, #cir.const_array<[#cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i]> : !cir.array<!u8i x 3>, #cir.int<42> : !s32i}> : !rec_anon_struct1 {alignment = 4 : i64}
// CHECK-DAG: g10 = #cir.global_view<@g9, [2 : i32]> : !cir.ptr<!s32i> {alignment = 8 : i64}
// CHECK-DAG: g11 = #cir.const_record<{#cir.int<1> : !s8i, #cir.int<42> : !s32i}> : !rec_S2 {alignment = 1 : i64}
// CHECK-DAG: g12 = #cir.global_view<@g11, [1 : i32]> : !cir.ptr<!s32i> {alignment = 8 : i64} 

// LLVM-DAG: @g9 = global { i8, [3 x i8], i32 } { i8 1, [3 x i8] zeroinitializer, i32 42 }, align 4
// LLVM-DAG: @g10 = global ptr getelementptr inbounds nuw (i8, ptr @g9, i64 4), align 8
// LLVM-DAG: @g11 = global %struct.S2 <{ i8 1, i32 42 }>, align 1
// LLVM-DAG: @g12 = global ptr getelementptr inbounds nuw (i8, ptr @g11, i64 1), align 8


typedef struct {
   short f0;
   int   f1;
   char  f2;
} S3;

static S3 g13 = {-1L,0L,1L};
static S3* g14[2][2] = {{0, &g13}, {&g13, &g13}};

// CHECK-DAG: g13 = #cir.const_record<{#cir.int<-1> : !s16i, #cir.const_array<[#cir.zero : !u8i, #cir.zero : !u8i]> : !cir.array<!u8i x 2>, #cir.int<0> : !s32i, #cir.int<1> : !s8i, #cir.const_array<[#cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i]> : !cir.array<!u8i x 3>}> : !rec_anon_struct3
// CHECK-DAG: g14 = #cir.const_array<[#cir.const_array<[#cir.ptr<null> : !cir.ptr<!rec_S3>, #cir.global_view<@g13> : !cir.ptr<!rec_S3>]> : !cir.array<!cir.ptr<!rec_S3> x 2>, #cir.const_array<[#cir.global_view<@g13> : !cir.ptr<!rec_S3>, #cir.global_view<@g13> : !cir.ptr<!rec_S3>]> : !cir.array<!cir.ptr<!rec_S3> x 2>]> : !cir.array<!cir.array<!cir.ptr<!rec_S3> x 2> x 2>

typedef struct {
   int  f0;
   int  f1;
} S4;

typedef struct {
   int f0 : 17;
   int f1 : 5;
   int f2 : 19;
   S4 f3;   
} S5;

static S5 g15 = {187,1,442,{123,321}};

int* g16 = &g15.f3.f1;

// CHECK-DAG: g15 = #cir.const_record<{#cir.int<187> : !u8i, #cir.int<0> : !u8i, #cir.int<2> : !u8i, #cir.zero : !u8i, #cir.int<186> : !u8i, #cir.int<1> : !u8i, #cir.int<0> : !u8i, #cir.zero : !u8i, #cir.const_record<{#cir.int<123> : !s32i, #cir.int<321> : !s32i}> : !rec_S4}> : !rec_anon_struct2 {alignment = 4 : i64}
// CHECK-DAG: g16 = #cir.global_view<@g15, [8, 1]> : !cir.ptr<!rec_anon_struct2> {alignment = 8 : i64}

// LLVM-DAG: @g15 = internal global { i8, i8, i8, i8, i8, i8, i8, i8, %struct.S4 } { i8 -69, i8 0, i8 2, i8 0, i8 -70, i8 1, i8 0, i8 0, %struct.S4 { i32 123, i32 321 } }, align 4
// LLVM-DAG: @g16 = global ptr getelementptr inbounds nuw (i8, ptr @g15, i64 12), align 8

void use() {
    int a = **g3;
    int b = ***g4; 
    int c = ****g5; 
    int d = **g8;
    S3 s = *g14[1][1];
    int f = *g16;
}
