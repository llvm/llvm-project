// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=BEFORE
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir -mmlir --mlir-print-ir-after=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=AFTER
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

void f1(__builtin_va_list c) {
  { __builtin_va_arg(c, void *); }
}

// BEFORE: cir.func @f1(%arg0: !ty_22__va_list22) attributes
// BEFORE: [[VAR_LIST:%.*]] = cir.alloca !ty_22__va_list22, !cir.ptr<!ty_22__va_list22>, ["c", init] {alignment = 8 : i64}
// BEFORE: cir.store %arg0, [[VAR_LIST]] : !ty_22__va_list22, !cir.ptr<!ty_22__va_list22>
// BEFORE: cir.scope {
// BEFORE-NEXT: [[TMP:%.*]] = cir.va.arg [[VAR_LIST]] : (!cir.ptr<!ty_22__va_list22>) -> !cir.ptr<!void>
// BEFORE-NEXT: }
// BEFORE-NEXT: cir.return

// AFTER: cir.func @f1(%arg0: !ty_22__va_list22) attributes
// AFTER: [[VARLIST:%.*]] = cir.alloca !ty_22__va_list22, !cir.ptr<!ty_22__va_list22>, ["c", init] {alignment = 8 : i64}
// AFTER: cir.store %arg0, [[VARLIST]] : !ty_22__va_list22, !cir.ptr<!ty_22__va_list22>
// AFTER: cir.scope {
//
// AFTER-NEXT: [[GR_OFFS_P:%.*]] = cir.get_member [[VARLIST]][3] {name = "gr_offs"} : !cir.ptr<!ty_22__va_list22> -> !cir.ptr<!s32i>
// AFTER-NEXT: [[GR_OFFS:%.*]] = cir.load [[GR_OFFS_P]] : !cir.ptr<!s32i>
// AFTER:  [[ZERO:%.*]] = cir.const #cir.int<0> : !s32i
// AFTER:  [[CMP0:%.*]] = cir.cmp(ge, [[GR_OFFS]], [[ZERO]]) : !s32i, !cir.bool
// AFTER-NEXT:  cir.brcond [[CMP0]] [[BB_ON_STACK:\^bb.*]], [[BB_MAY_REG:\^bb.*]]

// This BB calculates to see if it is possible to pass arg in register.
// AFTER: [[BB_MAY_REG]]:
// AFTER-NEXT: [[EIGHT:%.*]] = cir.const #cir.int<8> : !s32i
// AFTER-NEXT: [[NEW_REG_OFFS:%.*]] = cir.binop(add, [[GR_OFFS]], [[EIGHT]]) : !s32i
// AFTER-NEXT: cir.store [[NEW_REG_OFFS]], [[GR_OFFS_P]] : !s32i, !cir.ptr<!s32i>
// AFTER-NEXT: [[CMP1:%.*]] = cir.cmp(le, [[NEW_REG_OFFS]], [[ZERO]]) : !s32i, !cir.bool
// AFTER-NEXT: cir.brcond [[CMP1]] [[BB_IN_REG:\^bb.*]], [[BB_ON_STACK]]

// arg is passed in register.
// AFTER: [[BB_IN_REG]]:
// AFTER-NEXT: [[GR_TOP_P:%.*]] = cir.get_member [[VARLIST]][1] {name = "gr_top"} : !cir.ptr<!ty_22__va_list22> -> !cir.ptr<!cir.ptr<!void>>
// AFTER-NEXT: [[GR_TOP:%.*]] = cir.load [[GR_TOP_P]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// AFTER-NEXT: [[TMP2:%.*]] = cir.cast(bitcast, [[GR_TOP]] : !cir.ptr<!void>), !cir.ptr<i8>
// AFTER-NEXT: [[TMP3:%.*]] = cir.ptr_stride([[TMP2]] : !cir.ptr<i8>, [[GR_OFFS]] : !s32i), !cir.ptr<i8>
// AFTER-NEXT: [[IN_REG_OUTPUT:%.*]] = cir.cast(bitcast, [[TMP3]] : !cir.ptr<i8>), !cir.ptr<!void>
// AFTER-NEXT: cir.br [[BB_END:\^bb.*]]([[IN_REG_OUTPUT]] : !cir.ptr<!void>)

// arg is passed in stack.
// AFTER: [[BB_ON_STACK]]:
// AFTER-NEXT: [[STACK_P:%.*]] = cir.get_member [[VARLIST]][0] {name = "stack"} : !cir.ptr<!ty_22__va_list22> -> !cir.ptr<!cir.ptr<!void>>
// AFTER-NEXT: [[STACK_V:%.*]] = cir.load [[STACK_P]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// AFTER-NEXT: [[EIGHT_IN_PTR_ARITH:%.*]]  = cir.const #cir.int<8> : !u64i
// AFTER-NEXT: [[TMP4:%.*]] = cir.cast(bitcast, [[STACK_V]] : !cir.ptr<!void>), !cir.ptr<i8>
// AFTER-NEXT: [[TMP5:%.*]] = cir.ptr_stride([[TMP4]] : !cir.ptr<i8>, [[EIGHT_IN_PTR_ARITH]] : !u64i), !cir.ptr<i8>
// AFTER-NEXT: [[NEW_STACK_V:%.*]] = cir.cast(bitcast, [[TMP5]] : !cir.ptr<i8>), !cir.ptr<!void>
// AFTER-NEXT: cir.store [[NEW_STACK_V]], [[STACK_P]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// AFTER-NEXT: cir.br [[BB_END]]([[STACK_V]] : !cir.ptr<!void>)

// This BB is where different path converges. BLK_ARG is the arg addr which
// could come from IN_REG block where arg is passed in register, and saved in callee
// stack's argument saving area.
// Or from ON_STACK block which means arg is passed in from caller's stack area.
// AFTER-NEXT: [[BB_END]]([[BLK_ARG:%.*]]: !cir.ptr<!void>):  // 2 preds: [[BB_IN_REG]], [[BB_ON_STACK]]
// AFTER-NEXT:  [[TMP0:%.*]] = cir.cast(bitcast, [[BLK_ARG]] : !cir.ptr<!void>), !cir.ptr<!cir.ptr<!void>>
// AFTER-NEXT:  [[TMP1:%.*]] = cir.load [[TMP0]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// AFTER-NEXT: cir.yield
// AFTER-NEXT: }
// AFTER-NEXT: cir.return

// LLVM: %struct.__va_list = type { ptr, ptr, ptr, i32, i32 }
// LLVM: define void @f1(%struct.__va_list %0)
// LLVM: [[VARLIST:%.*]] = alloca %struct.__va_list, i64 1, align 8,
// LLVM: [[GR_OFFS_P:%.*]] = getelementptr %struct.__va_list, ptr [[VARLIST]], i32 0, i32 3
// LLVM: [[GR_OFFS:%.*]] = load i32, ptr [[GR_OFFS_P]], align 4,
// LLVM-NEXT: [[CMP0:%.*]] = icmp sge i32 [[GR_OFFS]], 0,
// LLVM-NEXT: br i1 [[CMP0]], label %[[BB_ON_STACK:.*]], label %[[BB_MAY_REG:.*]],

// LLVM:  [[BB_MAY_REG]]: ;
// LLVM: [[NEW_REG_OFFS:%.*]] = add i32 [[GR_OFFS]], 8,
// LLVM: store i32 [[NEW_REG_OFFS]], ptr [[GR_OFFS_P]], align 4,
// LLVM-NEXT: [[CMP1:%.*]] = icmp sle i32 [[NEW_REG_OFFS]], 0,
// LLVM-NEXT: br i1 [[CMP1]], label %[[BB_IN_REG:.*]], label %[[BB_ON_STACK]],

// LLVM:  [[BB_IN_REG]]: ;
// LLVM-NEXT: [[GR_TOP_P:%.*]] = getelementptr %struct.__va_list, ptr [[VARLIST]], i32 0, i32 1,
// LLVM-NEXT: [[GR_TOP:%.*]] = load ptr, ptr [[GR_TOP_P]], align 8,
// LLVM-NEXT: [[EXT64_GR_OFFS:%.*]] = sext i32 [[GR_OFFS]] to i64,
// LLVM-NEXT: [[IN_REG_OUTPUT:%.*]] = getelementptr i8, ptr [[GR_TOP]], i64 [[EXT64_GR_OFFS]],
// LLVM-NEXT: br label %[[BB_END:.*]],

// LLVM:  [[BB_ON_STACK]]: ;
// LLVM-NEXT: [[STACK_P:%.*]] = getelementptr %struct.__va_list, ptr [[VARLIST]], i32 0, i32 0,
// LLVM-NEXT: [[STACK_V:%.*]] = load ptr, ptr [[STACK_P]], align 8,
// LLVM-NEXT: [[NEW_STACK_V:%.*]] = getelementptr i8, ptr [[STACK_V]], i64 8,
// LLVM-NEXT: store ptr [[NEW_STACK_V]], ptr [[STACK_P]], align 8,
// LLVM-NEXT: br label %[[BB_END]],

// LLVM: [[BB_END]]: ; preds = %[[BB_ON_STACK]], %[[BB_IN_REG]]
// LLVM-NEXT: [[PHIP:%.*]] = phi ptr [ [[IN_REG_OUTPUT]], %[[BB_IN_REG]] ], [ [[STACK_V]], %[[BB_ON_STACK]] ]
// LLVM-NEXT: [[PHIV:%.*]] = load ptr, ptr [[PHIP]], align 8,
// LLVM:  ret void,
