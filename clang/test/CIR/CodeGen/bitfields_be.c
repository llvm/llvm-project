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
//CIR:   [[TMP3:%.*]] = cir.get_bitfield(#bfi_c, [[TMP2]] : !cir.ptr<!u32i>) -> !s32i

//LLVM: define dso_local i32 @init(ptr %0) {
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
