// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -mconstructor-aliases -fclangir -emit-cir-flat %s -o %t.flat.cir
// RUN: FileCheck --input-file=%t.flat.cir --check-prefix=CIR_FLAT %s

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
    // CIR_FLAT:   %[[EH:.*]] = cir.eh.inflight_exception
    // CIR_FLAT:   cir.br ^[[BB_INT_IDX_SEL:.*]] loc
    // CIR_FLAT: ^[[BB_INT_IDX_SEL]]:  // pred: ^[[LPAD]]
    // CIR_FLAT:   %[[SEL:.*]] = cir.eh.selector %[[EH]]
  } catch (int idx) {
    // CIR_FLAT:   %[[INT_IDX_ID:.*]] = cir.eh.typeid @_ZTIi
    // CIR_FLAT:   %[[MATCH_CASE_INT_IDX:.*]] = cir.cmp(eq, %[[SEL]], %[[INT_IDX_ID]]) : !u32i, !cir.bool
    // CIR_FLAT:   cir.brcond %[[MATCH_CASE_INT_IDX]] ^[[BB_INT_IDX_CATCH:.*]], ^[[BB_CHAR_MSG_SEL:.*]] loc
    // CIR_FLAT: ^[[BB_INT_IDX_CATCH]]:  // pred: ^[[BB_INT_IDX_SEL]]
    // CIR_FLAT:   %[[PARAM_INT_IDX:.*]] = cir.catch_param -> !cir.ptr<!s32i>
    // CIR_FLAT:   cir.const #cir.int<98>
    // CIR_FLAT:   cir.br ^[[AFTER_TRY]]
    z = 98;
    idx++;
  } catch (const char* msg) {
    // CIR_FLAT: ^[[BB_CHAR_MSG_SEL]]:  // pred: ^[[BB_INT_IDX_SEL]]
    // CIR_FLAT:   %[[CHAR_MSG_ID:.*]] = cir.eh.typeid @_ZTIPKc
    // CIR_FLAT:   %[[MATCH_CASE_CHAR_MSG:.*]] = cir.cmp(eq, %[[SEL]], %[[CHAR_MSG_ID]])
    // CIR_FLAT:   cir.brcond %[[MATCH_CASE_CHAR_MSG]] ^[[BB_CHAR_MSG_CATCH:.*]], ^[[BB_RESUME:.*]] loc
    // CIR_FLAT: ^[[BB_CHAR_MSG_CATCH]]:  // pred: ^[[BB_CHAR_MSG_SEL]]
    // CIR_FLAT:   %[[PARAM_CHAR_MSG:.*]] = cir.catch_param -> !cir.ptr<!s8i>
    // CIR_FLAT:   cir.const #cir.int<99> : !s32i
    // CIR_FLAT:   cir.br ^[[AFTER_TRY]] loc
    z = 99;
    (void)msg[0];
  }
  // CIR_FLAT: ^[[BB_RESUME]]:
  // CIR_FLAT:   cir.resume

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
