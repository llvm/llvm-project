// RUN: %clang_cc1  -triple aarch64_be-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1  -triple aarch64_be-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1  -triple aarch64_be-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

typedef struct {
    int a : 4;
    int b : 11;
    int c : 17;
} S;

// CIR:  !rec_S = !cir.record<struct "S" {!u32i}>
// LLVM: %struct.S = type { i32 }
// OGCG: %struct.S = type { i32 }
void def() {
  S s;
}
int init(S* s) {
  return s->c;
}

//CIR: cir.func dso_local @init
//CIR:   [[TMP0:%.*]] = cir.alloca !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>, ["s", init] {alignment = 8 : i64}
//CIR:   [[TMP1:%.*]] = cir.load align(8) [[TMP0]] : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
//CIR:   [[TMP2:%.*]] = cir.get_member [[TMP1]][0] {name = "c"} : !cir.ptr<!rec_S> -> !cir.ptr<!u32i>
//CIR:   [[TMP3:%.*]] = cir.get_bitfield align(4) (#bfi_c, [[TMP2]] : !cir.ptr<!u32i>) -> !s32i

//LLVM: define dso_local i32 @init(ptr %0){{.*}} {
//LLVM:   [[TMP0:%.*]] = alloca ptr, i64 1, align 8
//LLVM:   [[TMP1:%.*]] = alloca i32, i64 1, align 4
//LLVM:   [[TMP2:%.*]] = load ptr, ptr [[TMP0]], align 8
//LLVM:   [[TMP3:%.*]] = getelementptr %struct.S, ptr [[TMP2]], i32 0, i32 0
//LLVM:   [[TMP4:%.*]] = load i32, ptr [[TMP3]], align 4
//LLVM:   [[TMP5:%.*]] = shl i32 [[TMP4]], 15
//LLVM:   [[TMP6:%.*]] = ashr i32 [[TMP5]], 15

//OGCG: define dso_local i32 @init
//OGCG:   [[TMP0:%.*]] = alloca ptr, align 8
//OGCG:   [[TMP1:%.*]] = load ptr, ptr [[TMP0]], align 8
//OGCG:   [[TMP2:%.*]] = load i32, ptr [[TMP1]], align 4
//OGCG:   [[TMP3:%.*]] = shl i32 [[TMP2]], 15
//OGCG:   [[TMP4:%.*]] = ashr i32 [[TMP3]], 15


void load(S* s) {
    s->a = -4;
    s->b = 42;
    s->c = -12345;
}

// field 'a'
// CIR: cir.func dso_local @load
// CIR:    %[[PTR0:.*]] = cir.alloca !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>, ["s", init] {alignment = 8 : i64} loc(#loc35)
// CIR:    %[[CONST1:.*]] = cir.const #cir.int<4> : !s32i
// CIR:    %[[MIN1:.*]] = cir.unary(minus, %[[CONST1]]) nsw : !s32i, !s32i
// CIR:    %[[VAL0:.*]] = cir.load align(8) %[[PTR0]] : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR:    %[[GET0:.*]] = cir.get_member %[[VAL0]][0] {name = "a"} : !cir.ptr<!rec_S> -> !cir.ptr<!u32i>
// CIR:    %[[SET0:.*]] = cir.set_bitfield align(4) (#bfi_a, %[[GET0]] : !cir.ptr<!u32i>, %[[MIN1]] : !s32i) -> !s32i

// LLVM: define dso_local void @load{{.*}}{{.*}}
// LLVM:   %[[PTR0:.*]] = load ptr
// LLVM:   %[[GET0:.*]] = getelementptr %struct.S, ptr %[[PTR0]], i32 0, i32 0
// LLVM:   %[[VAL0:.*]] = load i32, ptr %[[GET0]], align 4
// LLVM:   %[[AND0:.*]] = and i32 %[[VAL0]], 268435455
// LLVM:   %[[OR0:.*]] = or i32 %[[AND0]], -1073741824
// LLVM:   store i32 %[[OR0]], ptr %[[GET0]], align 4

// OGCG: define dso_local void @load
// OGCG:   %[[PTR0:.*]] = load ptr
// OGCG:   %[[VAL0:.*]] = load i32, ptr %[[PTR0]], align 4
// OGCG:   %[[AND0:.*]] = and i32 %[[VAL0]], 268435455
// OGCG:   %[[OR0:.*]] = or i32 %[[AND0]], -1073741824
// OGCG:   store i32 %[[OR0]], ptr %[[PTR0]], align 4

// field 'b'
// CIR:    %[[CONST2:.*]] = cir.const #cir.int<42> : !s32i
// CIR:    %[[VAL1:.*]] = cir.load align(8) %[[PTR0]] : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR:    %[[GET1:.*]] = cir.get_member %[[VAL1]][0] {name = "b"} : !cir.ptr<!rec_S> -> !cir.ptr<!u32i>
// CIR:    %[[SET1:.*]] = cir.set_bitfield align(4) (#bfi_b, %[[GET1]] : !cir.ptr<!u32i>, %[[CONST2]] : !s32i) -> !s32i

// LLVM:  %[[PTR1:.*]] = load ptr
// LLVM:  %[[GET1:.*]] = getelementptr %struct.S, ptr %[[PTR1]], i32 0, i32 0
// LLVM:  %[[VAL1:.*]] = load i32, ptr %[[GET1]], align 4
// LLVM:  %[[AND1:.*]] = and i32 %[[VAL1]], -268304385
// LLVM:  %[[OR1:.*]] = or i32 %[[AND1]], 5505024
// LLVM:  store i32 %[[OR1]], ptr %[[GET1]], align 4

// OGCG:   %[[PTR1:.*]] = load ptr
// OGCG:   %[[VAL1:.*]] = load i32, ptr %[[PTR1]], align 4
// OGCG:   %[[AND1:.*]] = and i32 %[[VAL1]], -268304385
// OGCG:   %[[OR1:.*]] = or i32 %[[AND1]], 5505024
// OGCG:   store i32 %[[OR1]], ptr %[[PTR1]], align 4

// field 'c'
// CIR:    %[[CONST3:.*]] = cir.const #cir.int<12345> : !s32i
// CIR:    %[[MIN2:.*]] = cir.unary(minus, %[[CONST3]]) nsw : !s32i, !s32i
// CIR:    %[[VAL2:.*]] = cir.load align(8) %[[PTR0]] : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR:    %[[GET2:.*]] = cir.get_member %[[VAL2]][0] {name = "c"} : !cir.ptr<!rec_S> -> !cir.ptr<!u32i>
// CIR:    %[[SET2:.*]] = cir.set_bitfield align(4) (#bfi_c, %[[GET2]] : !cir.ptr<!u32i>, %[[MIN2]] : !s32i) -> !s32i

// LLVM:  %[[PTR2:.*]] = load ptr
// LLVM:  %[[GET2:.*]] = getelementptr %struct.S, ptr  %[[PTR2]], i32 0, i32 0
// LLVM:  %[[VAL2:.*]] = load i32, ptr %[[GET2]], align 4
// LLVM:  %[[AND2:.*]] = and i32 %[[VAL2]], -131072
// LLVM:  %[[OR2:.*]] = or i32 %[[AND2]], 118727
// LLVM:  store i32 %[[OR2]], ptr %[[GET2]], align 4

// OGCG:   %[[PTR2:.*]] = load ptr
// OGCG:   %[[VAL2:.*]] = load i32, ptr %[[PTR2]], align 4
// OGCG:   %[[AND2:.*]] = and i32 %[[VAL2]], -131072
// OGCG:   %[[OR2:.*]] = or i32 %[[AND2]], 118727
// OGCG:   store i32 %[[OR2]], ptr %[[PTR2]], align 4
