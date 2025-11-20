// RUN: %clang_cc1 -triple x86_64 -emit-llvm %s -o - -fwrapv -fsanitize=signed-integer-overflow | FileCheck %s --check-prefix=WRAPV
// RUN: %clang_cc1 -triple x86_64 -emit-llvm %s -o - -ftrapv | FileCheck %s --check-prefixes=BOTH-TRAP,TRAPV
// RUN: %clang_cc1 -triple x86_64 -emit-llvm %s -o - -fsanitize=signed-integer-overflow | FileCheck %s --check-prefixes=BOTH-TRAP,CATCH_UB
// COM: TODO: Support -ftrapv-handler.

extern int abs(int x);

int absi(int x) {
// WRAPV:      [[ABS:%.*]] = call i32 @llvm.abs.i32(i32 %0, i1 false)
// WRAPV-NEXT: ret i32 [[ABS]]
//
// BOTH-TRAP:       [[NEG:%.*]] = call { i32, i1 } @llvm.ssub.with.overflow.i32(i32 0, i32 [[X:%.*]])
// BOTH-TRAP:       [[NEGV:%.*]] = extractvalue { i32, i1 } [[NEG]], 0
// BOTH-TRAP:       [[OFL:%.*]] = extractvalue { i32, i1 } [[NEG]], 1
// BOTH-TRAP:       [[NOFL:%.*]] = xor i1 [[OFL]], true
// BOTH-TRAP:       br i1 [[NOFL]], label %[[CONT:.*]], label %[[TRAP:[a-zA-Z_.]*]]
// BOTH-TRAP:       [[TRAP]]:
// TRAPV-NEXT:        llvm.ubsantrap
// CATCH_UB:          @__ubsan_handle_negate_overflow
// BOTH-TRAP-NEXT:    unreachable
// BOTH-TRAP:       [[CONT]]:
// BOTH-TRAP-NEXT:    [[ABSCOND:%.*]] = icmp slt i32 [[X]], 0
// BOTH-TRAP-NEXT:    select i1 [[ABSCOND]], i32 [[NEGV]], i32 [[X]]
  return abs(x);
}

int babsi(int x) {
// WRAPV:      [[ABS:%.*]] = call i32 @llvm.abs.i32(i32 %0, i1 false)
// WRAPV-NEXT: ret i32 [[ABS]]
//
// BOTH-TRAP:       [[NEG:%.*]] = call { i32, i1 } @llvm.ssub.with.overflow.i32(i32 0, i32 [[X:%.*]])
// BOTH-TRAP:       [[NEGV:%.*]] = extractvalue { i32, i1 } [[NEG]], 0
// BOTH-TRAP:       [[OFL:%.*]] = extractvalue { i32, i1 } [[NEG]], 1
// BOTH-TRAP:       [[NOFL:%.*]] = xor i1 [[OFL]], true
// BOTH-TRAP:       br i1 [[NOFL]], label %[[CONT:.*]], label %[[TRAP:[a-zA-Z_.]*]]
// BOTH-TRAP:       [[TRAP]]:
// TRAPV-NEXT:        llvm.ubsantrap
// CATCH_UB:          @__ubsan_handle_negate_overflow
// BOTH-TRAP-NEXT:    unreachable
// BOTH-TRAP:       [[CONT]]:
// BOTH-TRAP-NEXT:    [[ABSCOND:%.*]] = icmp slt i32 [[X]], 0
// BOTH-TRAP-NEXT:    select i1 [[ABSCOND]], i32 [[NEGV]], i32 [[X]]
  return __builtin_abs(x);
}
