// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -mconstructor-aliases -fclangir -emit-cir-flat %s -o %t.flat.cir
// RUN: FileCheck --input-file=%t.flat.cir --check-prefix=CIR_FLAT %s

double division(int a, int b);

// CIR: cir.func @_Z2tcv()
// CIR_FLAT: cir.func @_Z2tcv()
unsigned long long tc() {
  int x = 50, y = 3;
  unsigned long long z;

  try {
    int a = 4;
    // CIR_FLAT:     cir.br ^bb1
    // CIR_FLAT: ^bb1:  // pred: ^bb0
    // CIR_FLAT:     cir.alloca !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>, ["msg"]
    // CIR_FLAT:     cir.alloca !s32i, !cir.ptr<!s32i>, ["idx"]
    // CIR_FLAT:     cir.br ^bb2
    // CIR_FLAT: ^bb2:  // pred: ^bb1
    // CIR_FLAT:     %[[EH_PTR:.*]] = cir.alloca !cir.ptr<!cir.eh.info>, !cir.ptr<!cir.ptr<!cir.eh.info>>, ["__exception_ptr"]
    // CIR_FLAT:     cir.try_call exception(%[[EH_PTR]]) @_Z8divisionii(
    z = division(x, y);
    a++;

    // CIR_FLAT:     %[[LOAD_EH_PTR:.*]] = cir.load %[[EH_PTR]] : !cir.ptr<!cir.ptr<!cir.eh.info>>, !cir.ptr<!cir.eh.info>
    // CIR_FLAT:     cir.br ^bb3(%[[LOAD_EH_PTR]] : !cir.ptr<!cir.eh.info>)
    // CIR_FLAT: ^bb3(%[[EH_ARG:.*]]: !cir.ptr<!cir.eh.info> loc(fused[#loc1, #loc2])):  // pred: ^bb2
    // CIR_FLAT:     cir.catch(%[[EH_ARG:.*]] : !cir.ptr<!cir.eh.info>, [
  } catch (int idx) {
    z = 98;
    idx++;
  } catch (const char* msg) {
    z = 99;
    (void)msg[0];
  }

  return z;
}

