// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir -pic-is-pie -pic-level 1 %s -o %t1.cir
// RUN: FileCheck --input-file=%t1.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-llvm -pic-is-pie -pic-level 1 %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

void foo(int i) {

}

int main() {
  foo(2);
  return 0;
}

// CIR: cir.func @foo(%arg0: !s32i
// CIR-NEXT:   [[TMP0:%.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init] {alignment = 4 : i64}
// CIR-NEXT:   cir.store %arg0, [[TMP0]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:   cir.return

// CIR: cir.func no_proto @main() -> !s32i
// CIR: [[TMP1:%.*]] = cir.const #cir.int<2> : !s32i
// CIR: cir.call @foo([[TMP1]]) : (!s32i) -> ()

// LLVM: define dso_local void @foo(i32 [[TMP3:%.*]])
// LLVM: [[ARG_STACK:%.*]] = alloca i32, i64 1, align 4,
// LLVM: store i32 [[TMP3]], ptr [[ARG_STACK]], align 4
// LLVM: ret void,

// LLVM: define dso_local i32 @main()
// LLVM: [[TMP4:%.*]] = alloca i32, i64 1, align 4,
// LLVM: call void @foo(i32 2),
// LLVM: store i32 0, ptr [[TMP4]], align 4
// LLVM: [[RET_VAL:%.*]] = load i32, ptr [[TMP4]], align 4
// LLVM: ret i32 [[RET_VAL]],
