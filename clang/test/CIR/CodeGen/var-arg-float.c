// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=BEFORE
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir -mmlir --mlir-print-ir-after=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=AFTER
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
#include <stdarg.h>

double f1(int n, ...) {
  va_list valist;
  va_start(valist, n);
  double res = va_arg(valist, double);
  va_end(valist);
  return res;
}

// BEFORE: !ty_22__va_list22 = !cir.struct<struct "__va_list" {!cir.ptr<!cir.void>, !cir.ptr<!cir.void>, !cir.ptr<!cir.void>, !cir.int<s, 32>, !cir.int<s, 32>}
// BEFORE:  cir.func @f1(%arg0: !s32i, ...) -> !cir.double
// BEFORE:  [[RETP:%.*]] = cir.alloca !cir.double, !cir.ptr<!cir.double>, ["__retval"]
// BEFORE:  [[RESP:%.*]] = cir.alloca !cir.double, !cir.ptr<!cir.double>, ["res", init]
// BEFORE:  cir.va.start [[VARLIST:%.*]] : !cir.ptr<!ty_22__va_list22>
// BEFORE:  [[TMP0:%.*]] = cir.va.arg [[VARLIST]] : (!cir.ptr<!ty_22__va_list22>) -> !cir.double
// BEFORE:  cir.store [[TMP0]], [[RESP]] : !cir.double, !cir.ptr<!cir.double>
// BEFORE:  cir.va.end [[VARLIST]] : !cir.ptr<!ty_22__va_list22>
// BEFORE:  [[RES:%.*]] = cir.load [[RESP]] : !cir.ptr<!cir.double>, !cir.double
// BEFORE:   cir.store [[RES]], [[RETP]] : !cir.double, !cir.ptr<!cir.double>
// BEFORE:  [[RETV:%.*]] = cir.load [[RETP]] : !cir.ptr<!cir.double>, !cir.double
// BEFORE:   cir.return [[RETV]] : !cir.double

// beginning block cir code
// AFTER: !ty_22__va_list22 = !cir.struct<struct "__va_list" {!cir.ptr<!cir.void>, !cir.ptr<!cir.void>, !cir.ptr<!cir.void>, !cir.int<s, 32>, !cir.int<s, 32>}
// AFTER:  cir.func @f1(%arg0: !s32i, ...) -> !cir.double
// AFTER:  [[RETP:%.*]] = cir.alloca !cir.double, !cir.ptr<!cir.double>, ["__retval"]
// AFTER:  [[RESP:%.*]] = cir.alloca !cir.double, !cir.ptr<!cir.double>, ["res", init]
// AFTER:  cir.va.start [[VARLIST:%.*]] : !cir.ptr<!ty_22__va_list22>
// AFTER:  [[VR_OFFS_P:%.*]] = cir.get_member [[VARLIST]][4] {name = "vr_offs"} : !cir.ptr<!ty_22__va_list22> -> !cir.ptr<!s32i>
// AFTER:  [[VR_OFFS:%.*]] = cir.load [[VR_OFFS_P]] : !cir.ptr<!s32i>, !s32i
// AFTER:  [[ZERO:%.*]] = cir.const #cir.int<0> : !s32i
// AFTER:  [[CMP0:%.*]] = cir.cmp(ge, [[VR_OFFS]], [[ZERO]]) : !s32i, !cir.bool
// AFTER-NEXT:  cir.brcond [[CMP0]] [[BB_ON_STACK:\^bb.*]], [[BB_MAY_REG:\^bb.*]]

// AFTER: [[BB_MAY_REG]]:
// AFTER-NEXT: [[SIXTEEN:%.*]] = cir.const #cir.int<16> : !s32i
// AFTER-NEXT: [[NEW_REG_OFFS:%.*]] = cir.binop(add, [[VR_OFFS]], [[SIXTEEN]]) : !s32i
// AFTER-NEXT: cir.store [[NEW_REG_OFFS]], [[VR_OFFS_P]] : !s32i, !cir.ptr<!s32i>
// AFTER-NEXT: [[CMP1:%.*]] = cir.cmp(le, [[NEW_REG_OFFS]], [[ZERO]]) : !s32i, !cir.bool
// AFTER-NEXT: cir.brcond [[CMP1]] [[BB_IN_REG:\^bb.*]], [[BB_ON_STACK]]


// AFTER: [[BB_IN_REG]]:
// AFTER-NEXT: [[VR_TOP_P:%.*]] = cir.get_member [[VARLIST]][2] {name = "vr_top"} : !cir.ptr<!ty_22__va_list22> -> !cir.ptr<!cir.ptr<!void>>
// AFTER-NEXT: [[VR_TOP:%.*]] = cir.load [[VR_TOP_P]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// AFTER-NEXT: [[TMP2:%.*]] = cir.cast(bitcast, [[VR_TOP]] : !cir.ptr<!void>), !cir.ptr<i8>
// AFTER-NEXT: [[TMP3:%.*]] = cir.ptr_stride([[TMP2]] : !cir.ptr<i8>, [[VR_OFFS]] : !s32i), !cir.ptr<i8>
// AFTER-NEXT: [[IN_REG_OUTPUT:%.*]] = cir.cast(bitcast, [[TMP3]] : !cir.ptr<i8>), !cir.ptr<!void>
// AFTER-NEXT: cir.br [[BB_END:\^bb.*]]([[IN_REG_OUTPUT]] : !cir.ptr<!void>)


// AFTER: [[BB_ON_STACK]]:
// AFTER-NEXT: [[STACK_P:%.*]] = cir.get_member [[VARLIST]][0] {name = "stack"} : !cir.ptr<!ty_22__va_list22> -> !cir.ptr<!cir.ptr<!void>>
// AFTER-NEXT: [[STACK_V:%.*]] = cir.load [[STACK_P]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// AFTER-NEXT: [[EIGHT_IN_PTR_ARITH:%.*]]  = cir.const #cir.int<8> : !u64i
// AFTER-NEXT: [[TMP4:%.*]] = cir.cast(bitcast, [[STACK_V]] : !cir.ptr<!void>), !cir.ptr<i8>
// AFTER-NEXT: [[TMP5:%.*]] = cir.ptr_stride([[TMP4]] : !cir.ptr<i8>, [[EIGHT_IN_PTR_ARITH]] : !u64i), !cir.ptr<i8>
// AFTER-NEXT: [[NEW_STACK_V:%.*]] = cir.cast(bitcast, [[TMP5]] : !cir.ptr<i8>), !cir.ptr<!void>
// AFTER-NEXT: cir.store [[NEW_STACK_V]], [[STACK_P]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// AFTER-NEXT: cir.br [[BB_END]]([[STACK_V]] : !cir.ptr<!void>)

// AFTER-NEXT: [[BB_END]]([[BLK_ARG:%.*]]: !cir.ptr<!void>):  // 2 preds: [[BB_IN_REG]], [[BB_ON_STACK]]
// AFTER-NEXT:  [[TMP0:%.*]] = cir.cast(bitcast, [[BLK_ARG]] : !cir.ptr<!void>), !cir.ptr<!cir.double>
// AFTER-NEXT:  [[TMP1:%.*]] = cir.load [[TMP0]] : !cir.ptr<!cir.double>, !cir.double
// AFTER:   cir.store [[TMP1]], [[RESP]] : !cir.double, !cir.ptr<!cir.double>
// AFTER:   cir.va.end [[VARLIST]] : !cir.ptr<!ty_22__va_list22>
// AFTER:   [[RES:%.*]] = cir.load [[RESP]] : !cir.ptr<!cir.double>, !cir.double
// AFTER:   cir.store [[RES]], [[RETP]] : !cir.double, !cir.ptr<!cir.double>
// AFTER:  [[RETV:%.*]] = cir.load [[RETP]] : !cir.ptr<!cir.double>, !cir.double
// AFTER:   cir.return [[RETV]] : !cir.double

// beginning block llvm code
// LLVM: %struct.__va_list = type { ptr, ptr, ptr, i32, i32 }
// LLVM: define dso_local double @f1(i32 %0, ...)
// LLVM: [[ARGN:%.*]] = alloca i32, i64 1, align 4,
// LLVM: [[RETP:%.*]] = alloca double, i64 1, align 8,
// LLVM: [[RESP:%.*]] = alloca double, i64 1, align 8,
// LLVM: call void @llvm.va_start.p0(ptr [[VARLIST:%.*]]),
// LLVM: [[VR_OFFS_P:%.*]] = getelementptr %struct.__va_list, ptr [[VARLIST]], i32 0, i32 4
// LLVM: [[VR_OFFS:%.*]] = load i32, ptr [[VR_OFFS_P]], align 4,
// LLVM-NEXT: [[CMP0:%.*]] = icmp sge i32 [[VR_OFFS]], 0,
// LLVM-NEXT: br i1 [[CMP0]], label %[[BB_ON_STACK:.*]], label %[[BB_MAY_REG:.*]],

// LLVM:  [[BB_MAY_REG]]: ;
// LLVM-NEXT: [[NEW_REG_OFFS:%.*]] = add i32 [[VR_OFFS]], 16,
// LLVM-NEXT: store i32 [[NEW_REG_OFFS]], ptr [[VR_OFFS_P]], align 4,
// LLVM-NEXT: [[CMP1:%.*]] = icmp sle i32 [[NEW_REG_OFFS]], 0,
// LLVM-NEXT: br i1 [[CMP1]], label %[[BB_IN_REG:.*]], label %[[BB_ON_STACK]],

// LLVM:  [[BB_IN_REG]]: ;
// LLVM-NEXT: [[VR_TOP_P:%.*]] = getelementptr %struct.__va_list, ptr [[VARLIST]], i32 0, i32 2,
// LLVM-NEXT: [[VR_TOP:%.*]] = load ptr, ptr [[VR_TOP_P]], align 8,
// LLVM-NEXT: [[EXT64_VR_OFFS:%.*]] = sext i32 [[VR_OFFS]] to i64,
// LLVM-NEXT: [[IN_REG_OUTPUT:%.*]] = getelementptr i8, ptr [[VR_TOP]], i64 [[EXT64_VR_OFFS]],
// LLVM-NEXT: br label %[[BB_END:.*]],

// LLVM:  [[BB_ON_STACK]]: ;
// LLVM-NEXT: [[STACK_P:%.*]] = getelementptr %struct.__va_list, ptr [[VARLIST]], i32 0, i32 0,
// LLVM-NEXT: [[STACK_V:%.*]] = load ptr, ptr [[STACK_P]], align 8,
// LLVM-NEXT: [[NEW_STACK_V:%.*]] = getelementptr i8, ptr [[STACK_V]], i64 8,
// LLVM-NEXT: store ptr [[NEW_STACK_V]], ptr [[STACK_P]], align 8,
// LLVM-NEXT: br label %[[BB_END]],

// LLVM: [[BB_END]]: ; preds = %[[BB_ON_STACK]], %[[BB_IN_REG]]
// LLVM-NEXT: [[PHIP:%.*]] = phi ptr [ [[IN_REG_OUTPUT]], %[[BB_IN_REG]] ], [ [[STACK_V]], %[[BB_ON_STACK]] ]
// LLVM-NEXT: [[PHIV:%.*]] = load double, ptr [[PHIP]], align 8,
// LLVM-NEXT: store double [[PHIV]], ptr [[RESP]], align 8,
// LLVM: call void @llvm.va_end.p0(ptr [[VARLIST]]),
// LLVM: [[RES:%.*]] = load double, ptr [[RESP]], align 8,
// LLVM: store double [[RES]], ptr [[RETP]], align 8,
// LLVM: [[RETV:%.*]] = load double, ptr [[RETP]], align 8,
// LLVM-NEXT: ret double [[RETV]],
