// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -mconstructor-aliases -fclangir -emit-cir-flat %s -o %t.flat.cir
// RUN: FileCheck --input-file=%t.flat.cir --check-prefix=CIR_FLAT %s
// RUN_DISABLED: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -mconstructor-aliases -fclangir -emit-llvm %s -o %t.ll
// RUN_DISABLED: FileCheck --input-file=%t.flat.cir --check-prefix=CIR_LLVM %s
double division(int a, int b);

// CIR: cir.func @_Z2tcv()
// CIR_FLAT: cir.func @_Z2tcv()
unsigned long long tc() {
  int x = 50, y = 3;
  unsigned long long z;

  // CIR_FLAT:     cir.alloca !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>, ["msg"]
  // CIR_FLAT:     cir.alloca !s32i, !cir.ptr<!s32i>, ["idx"]
  // CIR_FLAT:     cir.br ^bb2
  try {
    // CIR_FLAT:   ^bb2:  // pred: ^bb1
    // CIR_FLAT:     cir.alloca !s32i, !cir.ptr<!s32i>
    // CIR_FLAT:     cir.try_call @_Z8divisionii({{.*}}) ^[[CONT:.*]], ^[[LPAD:.*]] : (!s32i, !s32i)
    int a = 4;
    z = division(x, y);

    // CIR_FLAT: ^[[CONT:.*]]:  // pred: ^bb2
    // CIR_FLAT: cir.cast(float_to_int, %12 : !cir.double), !u64i
    a++;
    // CIR_FLAT: cir.br ^[[AFTER_TRY:.*]] loc

    // CIR_FLAT: ^[[LPAD]]:  // pred: ^bb2
    // CIR_FLAT:   %[[EH:.*]], %[[SEL:.*]] = cir.eh.inflight_exception [@_ZTIi, @_ZTIPKc]
    // CIR_FLAT:   cir.br ^[[BB_INT_IDX_SEL:.*]](%[[EH]], %[[SEL]] : {{.*}}) loc
  } catch (int idx) {
    // CIR_FLAT: ^[[BB_INT_IDX_SEL]](%[[INT_IDX_EH:.*]]: !cir.ptr<!void> loc({{.*}}), %[[INT_IDX_SEL:.*]]: !u32i
    // CIR_FLAT:   %[[INT_IDX_ID:.*]] = cir.eh.typeid @_ZTIi
    // CIR_FLAT:   %[[MATCH_CASE_INT_IDX:.*]] = cir.cmp(eq, %[[INT_IDX_SEL]], %[[INT_IDX_ID]]) : !u32i, !cir.bool
    // CIR_FLAT:   cir.brcond %[[MATCH_CASE_INT_IDX]] ^[[BB_INT_IDX_CATCH:.*]](%[[INT_IDX_EH]] : {{.*}}), ^[[BB_CHAR_MSG_CMP:.*]](%[[INT_IDX_EH]], %[[INT_IDX_SEL]] : {{.*}}) loc
    // CIR_FLAT: ^[[BB_INT_IDX_CATCH]](%[[INT_IDX_CATCH_SLOT:.*]]: !cir.ptr<!void>
    // CIR_FLAT:   %[[PARAM_INT_IDX:.*]] = cir.catch_param begin %[[INT_IDX_CATCH_SLOT]] -> !cir.ptr<!s32i>
    // CIR_FLAT:   cir.const #cir.int<98>
    // CIR_FLAT:   cir.br ^[[AFTER_TRY]] loc
    z = 98;
    idx++;
  } catch (const char* msg) {
    // CIR_FLAT: ^[[BB_CHAR_MSG_CMP]](%[[CHAR_MSG_EH:.*]]: !cir.ptr<!void> loc({{.*}}), %[[CHAR_MSG_SEL:.*]]: !u32i
    // CIR_FLAT:   %[[CHAR_MSG_ID:.*]] = cir.eh.typeid @_ZTIPKc
    // CIR_FLAT:   %[[MATCH_CASE_CHAR_MSG:.*]] = cir.cmp(eq, %[[CHAR_MSG_SEL]], %[[CHAR_MSG_ID]])
    // CIR_FLAT:   cir.brcond %[[MATCH_CASE_CHAR_MSG]] ^[[BB_CHAR_MSG_CATCH:.*]](%[[CHAR_MSG_EH]] : {{.*}}), ^[[BB_RESUME:.*]](%[[CHAR_MSG_EH]], %[[CHAR_MSG_SEL]] : {{.*}}) loc
    // CIR_FLAT: ^[[BB_CHAR_MSG_CATCH]](%[[CHAR_MSG_CATCH_SLOT:.*]]: !cir.ptr<!void>
    // CIR_FLAT:   %[[PARAM_CHAR_MSG:.*]] = cir.catch_param begin %[[CHAR_MSG_CATCH_SLOT]] -> !cir.ptr<!s8i>
    // CIR_FLAT:   cir.const #cir.int<99> : !s32i
    // CIR_FLAT:   cir.br ^[[AFTER_TRY]] loc
    z = 99;
    (void)msg[0];
  }
  // CIR_FLAT: ^[[BB_RESUME]](%[[RESUME_EH:.*]]: !cir.ptr<!void> loc({{.*}}), %[[RESUME_SEL:.*]]: !u32i
  // CIR_FLAT:   cir.resume %[[RESUME_EH]], %[[RESUME_SEL]]

  // CIR_FLAT: ^[[AFTER_TRY]]:
  // CIR_FLAT: cir.load

  return z;
}

// CIR_FLAT: cir.func @_Z3tc2v
unsigned long long tc2() {
  int x = 50, y = 3;
  unsigned long long z;

  try {
    int a = 4;
    z = division(x, y);
    a++;
  } catch (int idx) {
    // CIR_FLAT: cir.eh.inflight_exception [@_ZTIi, @_ZTIPKc]
    z = 98;
    idx++;
  } catch (const char* msg) {
    z = 99;
    (void)msg[0];
  } catch (...) {
    // CIR_FLAT:   cir.catch_param
    // CIR_FLAT:   cir.const #cir.int<100> : !s32i
    // CIR_FLAT:   cir.br ^[[AFTER_TRY:.*]] loc
    // CIR_FLAT: ^[[AFTER_TRY]]:  // 4 preds
    // CIR_FLAT:   cir.load
    // CIR_FLAT:   cir.return
    z = 100;
  }

  return z;
}

// CIR_FLAT: cir.func @_Z3tc3v
unsigned long long tc3() {
  int x = 50, y = 3;
  unsigned long long z;

  try {
    z = division(x, y);
  } catch (...) {
    // CIR_FLAT:   cir.eh.inflight_exception loc
    // CIR_FLAT:   cir.br ^[[CATCH_ALL:.*]]({{.*}} : {{.*}}) loc
    // CIR_FLAT: ^[[CATCH_ALL]](%[[CATCH_ALL_EH:.*]]: !cir.ptr<!void>
    // CIR_FLAT:   cir.catch_param begin %[[CATCH_ALL_EH]] -> !cir.ptr<!void>
    // CIR_FLAT:   cir.const #cir.int<100> : !s32i
    // CIR_FLAT:   cir.br ^[[AFTER_TRY:.*]] loc
    // CIR_FLAT: ^[[AFTER_TRY]]:  // 2 preds
    // CIR_FLAT:   cir.load
    z = 100;
  }

  return z;
}