// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck %s --check-prefix=CIR --input-file=%t.cir

auto func() -> int {
  return __builtin_strcmp("", "");
  // CIR:      cir.func {{.*}} @_Z4funcv()
  // CIR-NEXT: %[[RET_VAL:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
  // CIR-NEXT: %[[VAL:.*]] = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: cir.store{{.*}} %[[VAL]], %[[RET_VAL]] : !s32i, !cir.ptr<!s32i>
  // CIR-NEXT: %[[TMP:.*]] = cir.load{{.*}} %0 : !cir.ptr<!s32i>, !s32i
  // CIR-NEXT: cir.return %[[TMP]] : !s32i
}

auto func2() -> int {
  return __builtin_choose_expr(true, 1, 2);

  // CIR:      cir.func {{.*}} @_Z5func2v()
  // CIR-NEXT:   %[[RET_VAL:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
  // CIR-NEXT:   %[[VAL:.*]] = cir.const #cir.int<1> : !s32i
  // CIR-NEXT:   cir.store{{.*}} %[[VAL]], %[[RET_VAL]] : !s32i, !cir.ptr<!s32i>
  // CIR-NEXT:   %[[TMP:.*]] = cir.load{{.*}} %[[RET_VAL]] : !cir.ptr<!s32i>, !s32i
  // CIR-NEXT:   cir.return %[[TMP]] : !s32i
}

auto func3() -> int {
  return __builtin_choose_expr(false, 1, 2);

  // CIR:      cir.func {{.*}} @_Z5func3v()
  // CIR-NEXT:   %[[RET_VAL:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
  // CIR-NEXT:   %[[VAL:.*]] = cir.const #cir.int<2> : !s32i
  // CIR-NEXT:   cir.store{{.*}} %[[VAL]], %[[RET_VAL]] : !s32i, !cir.ptr<!s32i>
  // CIR-NEXT:   %[[TMP:.*]] = cir.load{{.*}} %[[RET_VAL]] : !cir.ptr<!s32i>, !s32i
  // CIR-NEXT:   cir.return %[[TMP]] : !s32i
}
