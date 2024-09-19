// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-canonicalize %s -o %t1.cir 2>&1 | FileCheck -check-prefix=CIR-BEFORE %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -fclangir -emit-cir -mmlir --mlir-print-ir-after=cir-simplify %s -o %t2.cir 2>&1 | FileCheck -check-prefix=CIR-AFTER %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -fclangir -emit-llvm %s -o %t.ll 
// RUN: FileCheck --input-file=%t.ll --check-prefix=LLVM %s

int test(bool x) {
  return x ? 1 : 2;
}

//      CIR-BEFORE: cir.func @_Z4testb
//      CIR-BEFORE:   %{{.+}} = cir.ternary(%{{.+}}, true {
// CIR-BEFORE-NEXT:     %[[#A:]] = cir.const #cir.int<1> : !s32i
// CIR-BEFORE-NEXT:     cir.yield %[[#A]] : !s32i
// CIR-BEFORE-NEXT:   }, false {
// CIR-BEFORE-NEXT:     %[[#B:]] = cir.const #cir.int<2> : !s32i
// CIR-BEFORE-NEXT:     cir.yield %[[#B]] : !s32i
// CIR-BEFORE-NEXT:   }) : (!cir.bool) -> !s32i
//      CIR-BEFORE: }

//      CIR-AFTER: cir.func @_Z4testb
//      CIR-AFTER:   %[[#A:]] = cir.const #cir.int<1> : !s32i
// CIR-AFTER-NEXT:   %[[#B:]] = cir.const #cir.int<2> : !s32i
// CIR-AFTER-NEXT:   %{{.+}} = cir.select if %{{.+}} then %[[#A]] else %[[#B]] : (!cir.bool, !s32i, !s32i) -> !s32i
//      CIR-AFTER: }

// LLVM: @_Z4testb
// LLVM:   %{{.+}} = select i1 %{{.+}}, i32 1, i32 2
// LLVM: }

int test2(bool cond) {
  constexpr int x = 1;
  constexpr int y = 2;
  return cond ? x : y;
}

//      CIR-BEFORE: cir.func  @_Z5test2b
//      CIR-BEFORE:   %[[#COND:]] = cir.load %{{.+}} : !cir.ptr<!cir.bool>, !cir.bool
// CIR-BEFORE-NEXT:   %{{.+}} = cir.ternary(%[[#COND]], true {
// CIR-BEFORE-NEXT:     %[[#A:]] = cir.const #cir.int<1> : !s32i
// CIR-BEFORE-NEXT:     cir.yield %[[#A]] : !s32i
// CIR-BEFORE-NEXT:   }, false {
// CIR-BEFORE-NEXT:     %[[#B:]] = cir.const #cir.int<2> : !s32i
// CIR-BEFORE-NEXT:     cir.yield %[[#B]] : !s32i
// CIR-BEFORE-NEXT:   }) : (!cir.bool) -> !s32i
//      CIR-BEFORE: }

//      CIR-AFTER: cir.func @_Z5test2b
//      CIR-AFTER:   %[[#COND:]] = cir.load %{{.+}} : !cir.ptr<!cir.bool>, !cir.bool
// CIR-AFTER-NEXT:   %[[#A:]] = cir.const #cir.int<1> : !s32i
// CIR-AFTER-NEXT:   %[[#B:]] = cir.const #cir.int<2> : !s32i
// CIR-AFTER-NEXT:   %{{.+}} = cir.select if %[[#COND]] then %[[#A]] else %[[#B]] : (!cir.bool, !s32i, !s32i) -> !s32i
//      CIR-AFTER: }

// LLVM: @_Z5test2b
// LLVM:   %{{.+}} = select i1 %{{.+}}, i32 1, i32 2
// LLVM: }
