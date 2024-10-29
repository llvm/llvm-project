// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

void sw1(int a) {
  switch (int b = 1; a) {
  case 0:
    b = b + 1;
    break;
  case 1:
    break;
  case 2: {
    b = b + 1;
    int yolo = 100;
    break;
  }
  }
}
// CHECK: cir.func @_Z3sw1i
// CHECK: cir.switch (%3 : !s32i) {
// CHECK-NEXT: cir.case(equal, [#cir.int<0> : !s32i]) {
// CHECK: cir.break
// CHECK: cir.case(equal, [#cir.int<1> : !s32i]) {
// CHECK-NEXT: cir.break
// CHECK: cir.case(equal, [#cir.int<2> : !s32i]) {
// CHECK: cir.scope {
// CHECK: cir.alloca !s32i, !cir.ptr<!s32i>, ["yolo", init]
// CHECK: cir.break

void sw2(int a) {
  switch (int yolo = 2; a) {
  case 3:
    // "fomo" has the same lifetime as "yolo"
    int fomo = 0;
    yolo = yolo + fomo;
    break;
  }
}

// CHECK: cir.func @_Z3sw2i
// CHECK: cir.scope {
// CHECK-NEXT:   %1 = cir.alloca !s32i, !cir.ptr<!s32i>, ["yolo", init]
// CHECK-NEXT:   %2 = cir.alloca !s32i, !cir.ptr<!s32i>, ["fomo", init]
// CHECK:        cir.switch (%4 : !s32i) {
// CHECK-NEXT:   cir.case(equal, [#cir.int<3> : !s32i]) {
// CHECK-NEXT:     %5 = cir.const #cir.int<0> : !s32i
// CHECK-NEXT:     cir.store %5, %2 : !s32i, !cir.ptr<!s32i>

void sw3(int a) {
  switch (a) {
  default:
    break;
  }
}

// CHECK: cir.func @_Z3sw3i
// CHECK: cir.scope {
// CHECK-NEXT:   %1 = cir.load %0 : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT:   cir.switch (%1 : !s32i) {
// CHECK-NEXT:   cir.case(default, []) {
// CHECK-NEXT:     cir.break
// CHECK-NEXT:   }
// CHECK-NEXT:   cir.yield
// CHECK-NEXT:   }

int sw4(int a) {
  switch (a) {
  case 42: {
    return 3;
  }
  default:
    return 2;
  }
  return 0;
}

// CHECK: cir.func @_Z3sw4i
// CHECK:       cir.switch (%4 : !s32i) {
// CHECK-NEXT:       cir.case(equal, [#cir.int<42> : !s32i]) {
// CHECK-NEXT:         cir.scope {
// CHECK-NEXT:           %5 = cir.const #cir.int<3> : !s32i
// CHECK-NEXT:           cir.store %5, %1 : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT:           %6 = cir.load %1 : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT:           cir.return %6 : !s32i
// CHECK-NEXT:         }
// CHECK-NEXT:         cir.yield
// CHECK-NEXT:       }
// CHECK-NEXT:       cir.case(default, []) {
// CHECK-NEXT:         %5 = cir.const #cir.int<2> : !s32i
// CHECK-NEXT:         cir.store %5, %1 : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT:         %6 = cir.load %1 : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT:         cir.return %6 : !s32i
// CHECK-NEXT:       }
// CHECK-NEXT:       cir.yield
// CHECK-NEXT:       }

void sw5(int a) {
  switch (a) {
  case 1:;
  }
}

// CHECK: cir.func @_Z3sw5i
// CHECK: cir.switch (%1 : !s32i) {
// CHECK-NEXT:   cir.case(equal, [#cir.int<1> : !s32i]) {
// CHECK-NEXT:     cir.yield
// CHECK-NEXT:   }
// CHECK-NEXT:   cir.yield
// CHECK-NEXT:   }

void sw6(int a) {
  switch (a) {
  case 0:
  case 1:
  case 2:
    break;
  case 3:
  case 4:
  case 5:
    break;
  }
}

// CHECK: cir.func @_Z3sw6i
// CHECK: cir.switch (%1 : !s32i) {
// CHECK-NEXT: cir.case(anyof, [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i]) {
// CHECK-NEXT:   cir.break
// CHECK-NEXT: }
// CHECK-NEXT: cir.case(anyof, [#cir.int<3> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i]) {
// CHECK-NEXT:   cir.break
// CHECK-NEXT: }

void sw7(int a) {
  switch (a) {
  case 0:
  case 1:
  case 2:
    int x;
  case 3:
  case 4:
  case 5:
    break;
  }
}

// CHECK: cir.func @_Z3sw7i
// CHECK: cir.case(anyof, [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i]) {
// CHECK-NEXT:   cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: cir.case(anyof, [#cir.int<3> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i]) {
// CHECK-NEXT:   cir.break
// CHECK-NEXT: }

void sw8(int a) {
  switch (a)
  {
  case 3:
    break;
  case 4:
  default:
    break;
  }
}

//CHECK:    cir.func @_Z3sw8i
//CHECK:      cir.case(equal, [#cir.int<3> : !s32i]) {
//CHECK-NEXT:   cir.break
//CHECK-NEXT: }
//CHECK-NEXT: cir.case(equal, [#cir.int<4> : !s32i]) {
//CHECK-NEXT:   cir.yield
//CHECK-NEXT: }
//CHECK-NEXT: cir.case(default, []) {
//CHECK-NEXT:   cir.break
//CHECK-NEXT: }

void sw9(int a) {
  switch (a)
  {
  case 3:
    break;
  default:
  case 4:
    break;
  }
}

//CHECK:    cir.func @_Z3sw9i
//CHECK:      cir.case(equal, [#cir.int<3> : !s32i]) {
//CHECK-NEXT:   cir.break
//CHECK-NEXT: }
//CHECK-NEXT: cir.case(default, []) {
//CHECK-NEXT:   cir.yield
//CHECK-NEXT: }
//CHECK-NEXT: cir.case(equal, [#cir.int<4> : !s32i]) {
//CHECK-NEXT:   cir.break
//CHECK-NEXT: }

void sw10(int a) {
  switch (a)
  {
  case 3:
    break;
  case 4:
  default:
  case 5:
    break;
  }
}

//CHECK:    cir.func @_Z4sw10i
//CHECK:      cir.case(equal, [#cir.int<3> : !s32i]) {
//CHECK-NEXT:   cir.break
//CHECK-NEXT: }
//CHECK-NEXT: cir.case(equal, [#cir.int<4> : !s32i]) {
//CHECK-NEXT:   cir.yield
//CHECK-NEXT: }
//CHECK-NEXT: cir.case(default, []) {
//CHECK-NEXT:   cir.yield
//CHECK-NEXT: }
//CHECK-NEXT: cir.case(equal, [#cir.int<5> : !s32i]) {
//CHECK-NEXT:   cir.break
//CHECK-NEXT: }

void sw11(int a) {
  switch (a)
  {
  case 3:
    break;
  case 4:
  case 5:
  default:
  case 6:
  case 7:
    break;
  }
}

//CHECK:    cir.func @_Z4sw11i
//CHECK:      cir.case(equal, [#cir.int<3> : !s32i]) {
//CHECK-NEXT:   cir.break
//CHECK-NEXT: }
//CHECK-NEXT: cir.case(anyof, [#cir.int<4> : !s32i, #cir.int<5> : !s32i]) {
//CHECK-NEXT:   cir.yield
//CHECK-NEXT: }
//CHECK-NEXT: cir.case(default, []) {
//CHECK-NEXT:   cir.yield
//CHECK-NEXT: }
//CHECK-NEXT: cir.case(anyof, [#cir.int<6> : !s32i, #cir.int<7> : !s32i]) {
//CHECK-NEXT:   cir.break
//CHECK-NEXT: }

void sw12(int a) {
  switch (a)
  {
  case 3:
    return;
    break;
  }
}

//      CHECK: cir.func @_Z4sw12i
//      CHECK:   cir.scope {
//      CHECK:     cir.switch
// CHECK-NEXT:     cir.case(equal, [#cir.int<3> : !s32i]) {
// CHECK-NEXT:       cir.return
// CHECK-NEXT:     ^bb1:  // no predecessors
// CHECK-NEXT:       cir.break
// CHECK-NEXT:     }

void sw13(int a, int b) {
  switch (a) {
  case 1:
    switch (b) {
    case 2:
      break;
    }
  }
}

//      CHECK:  cir.func @_Z4sw13ii
//      CHECK:    cir.scope {
//      CHECK:      cir.switch
// CHECK-NEXT:      cir.case(equal, [#cir.int<1> : !s32i]) {
// CHECK-NEXT:        cir.scope {
//      CHECK:          cir.switch
// CHECK-NEXT:          cir.case(equal, [#cir.int<2> : !s32i]) {
// CHECK-NEXT:            cir.break
// CHECK-NEXT:          }
// CHECK-NEXT:          cir.yield
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK:         cir.yield
//      CHECK:    }
//      CHECK:    cir.return

void fallthrough(int x) {
  switch (x) {
    case 1:
      __attribute__((fallthrough));
    case 2:
      break;
    default:
      break;
  }
}

//      CHECK:  cir.func @_Z11fallthroughi
//      CHECK:    cir.scope {
//      CHECK:      cir.switch (%1 : !s32i) {
// CHECK-NEXT:      cir.case(equal, [#cir.int<1> : !s32i]) {
// CHECK-NEXT:        cir.yield
// CHECK-NEXT:      }
// CHECK-NEXT:      cir.case(equal, [#cir.int<2> : !s32i]) {
// CHECK-NEXT:        cir.break
// CHECK-NEXT:      }
// CHECK-NEXT:      cir.case(default, []) {
// CHECK-NEXT:        cir.break
// CHECK-NEXT:      }
// CHECK-NEXT:      cir.yield
// CHECK-NEXT:      }
// CHECK-NEXT:    }

int unreachable_after_break_1(int a) {
  switch (a) {
    case(42):
      break;
      goto exit;
    default:
      return 0;
  };

exit:
  return -1;

}
// CHECK: cir.func @_Z25unreachable_after_break_1i
// CHECK:   cir.case(equal, [#cir.int<42> : !s32i]) {
// CHECK:     cir.break
// CHECK:   ^bb1:  // no predecessors
// CHECK:     cir.goto "exit"
// CHECK:   }

int nested_switch(int a) {
  switch (int b = 1; a) {
  case 0:
    b = b + 1;
  case 1:
    return b;
  case 2: {
    b = b + 1;
    if (a > 1000) {
        case 9:
          b += a;
    }
    if (a > 500) {
        case 7:
          return a + b;
    }
    break;
  }
  }

  return 0;
}

// CHECK: cir.switch (%6 : !s32i) {
// CHECK:   cir.case(equal, [#cir.int<0> : !s32i]) {
// CHECK:     cir.yield
// CHECK:   }
// CHECK:   cir.case(equal, [#cir.int<1> : !s32i]) {
// CHECK:     cir.return
// CHECK:   }
// CHECK:   cir.case(equal, [#cir.int<2> : !s32i]) {
// CHECK:     cir.scope {
// CHECK:     cir.scope {
// CHECK:       cir.if
// CHECK:         cir.case(equal, [#cir.int<9> : !s32i]) {
// CHECK:         cir.yield
// CHECK:     cir.scope {
// CHECK:         cir.if
// CHECK:           cir.case(equal, [#cir.int<7> : !s32i]) {
// CHECK:           cir.return
