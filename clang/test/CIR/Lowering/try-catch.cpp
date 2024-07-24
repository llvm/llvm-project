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
    // CIR_FLAT_DISABLED:     cir.alloca !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>, ["msg"]
    // CIR_FLAT_DISABLED:     cir.alloca !s32i, !cir.ptr<!s32i>, ["idx"]
    // CIR_FLAT:     cir.br ^bb1
    // CIR_FLAT:   ^bb1:  // pred: ^bb0
    // CIR_FLAT:     cir.br ^bb2
    // CIR_FLAT:   ^bb2:  // pred: ^bb1
    // CIR_FLAT:     cir.call exception @_Z8divisionii(
    z = division(x, y);
    a++;

    // FIXME: this is temporary, should branch directly to ^bb4
    // but if done now it would be stripped by MLIR simplification.
    // CIR_FLAT:   cir.br ^bb3

    // CIR_FLAT: ^bb3:  // pred: ^bb2
    // CIR_FLAT:   %[[EH:.*]] = cir.eh.inflight_exception
    // CIR_FLAT:   cir.br ^bb4
    // CIR_FLAT: ^bb4:  // pred: ^bb3
    // CIR_FLAT:   %[[SEL:.*]] = cir.eh.selector %[[EH]]
  } catch (int idx) {
    z = 98;
    idx++;
  } catch (const char* msg) {
    z = 99;
    (void)msg[0];
  }

  return z;
}

