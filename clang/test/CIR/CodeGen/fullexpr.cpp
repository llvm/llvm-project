// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir-flat %s -o %t.cir.flat
// RUN: FileCheck --check-prefix=FLAT  --input-file=%t.cir.flat %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -o - %s \
// RUN: | opt -S -passes=instcombine,mem2reg,simplifycfg -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

int go(int const& val);

int go1() {
  auto x = go(1);
  return x;
}

// CHECK: cir.func @_Z3go1v() -> !s32i
// CHECK: %[[#XAddr:]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init] {alignment = 4 : i64}
// CHECK: %[[#RVal:]] = cir.scope {
// CHECK-NEXT:   %[[#TmpAddr:]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["ref.tmp0", init] {alignment = 4 : i64}
// CHECK-NEXT:   %[[#One:]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT:   cir.store %[[#One]], %[[#TmpAddr]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT:   %[[#RValTmp:]] = cir.call @_Z2goRKi(%[[#TmpAddr]]) : (!cir.ptr<!s32i>) -> !s32i
// CHECK-NEXT:   cir.yield %[[#RValTmp]] : !s32i
// CHECK-NEXT: }
// CHECK-NEXT: cir.store %[[#RVal]], %[[#XAddr]] : !s32i, !cir.ptr<!s32i>

// FLAT: cir.func @_Z3go1v() -> !s32i
// FLAT: %[[#TmpAddr:]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["ref.tmp0", init] {alignment = 4 : i64}
// FLAT: %[[#XAddr:]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init] {alignment = 4 : i64}
// FLAT: cir.br ^[[before_body:.*]]{{ loc.*}}
// FLAT-NEXT: ^[[before_body]]:  // pred: ^bb0
// FLAT-NEXT:   %[[#One:]] = cir.const #cir.int<1> : !s32i
// FLAT-NEXT:   cir.store %[[#One]], %[[#TmpAddr]] : !s32i, !cir.ptr<!s32i>
// FLAT-NEXT:   %[[#RValTmp:]] = cir.call @_Z2goRKi(%[[#TmpAddr]]) : (!cir.ptr<!s32i>) -> !s32i
// FLAT-NEXT:   cir.br ^[[continue_block:.*]](%[[#RValTmp]] : !s32i) {{loc.*}}
// FLAT-NEXT: ^[[continue_block]](%[[#BlkArgRval:]]: !s32i {{loc.*}}):  // pred: ^[[before_body]]
// FLAT-NEXT:   cir.store %[[#BlkArgRval]], %[[#XAddr]] : !s32i, !cir.ptr<!s32i>

// LLVM-LABEL: @_Z3go1v()
// LLVM-NEXT: %[[#TmpAddr:]] = alloca i32, i64 1, align 4
// LLVM: br label %[[before_body:[0-9]+]]
// LLVM: [[before_body]]:
// LLVM-NEXT: store i32 1, ptr %[[#TmpAddr]], align 4
// LLVM-NEXT: %[[#RValTmp:]] = call i32 @_Z2goRKi(ptr %[[#TmpAddr]])
// LLVM-NEXT: br label %[[continue_block:[0-9]+]] 

// LLVM: [[continue_block]]:
// LLVM-NEXT: [[PHI:%.*]] = phi i32 [ %[[#RValTmp]], %[[before_body]] ]
// LLVM: store i32 [[PHI]], ptr [[TMP0:%.*]], align 4
// LLVM: [[TMP1:%.*]] = load i32, ptr [[TMP0]], align 4
// LLVM: store i32 [[TMP1]], ptr [[TMP2:%.*]], align 4
// LLVM: [[TMP3:%.*]] = load i32, ptr [[TMP2]], align 4
// LLVM: ret i32 [[TMP3]]
