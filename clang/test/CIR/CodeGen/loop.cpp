// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

void l0() {
  for (;;) {
  }
}

// CHECK: cir.func @_Z2l0v
// CHECK: cir.for : cond {
// CHECK:   %[[#TRUE:]] = cir.const(#true) : !cir.bool
// CHECK:   cir.condition(%[[#TRUE]])

void l1() {
  int x = 0;
  for (int i = 0; i < 10; i = i + 1) {
    x = x + 1;
  }
}

// CHECK: cir.func @_Z2l1v
// CHECK: cir.for : cond {
// CHECK-NEXT:   %4 = cir.load %2 : cir.ptr <!s32i>, !s32i
// CHECK-NEXT:   %5 = cir.const(#cir.int<10> : !s32i) : !s32i
// CHECK-NEXT:   %6 = cir.cmp(lt, %4, %5) : !s32i, !cir.bool
// CHECK-NEXT:   cir.condition(%6)
// CHECK-NEXT: } body {
// CHECK-NEXT:   %4 = cir.load %0 : cir.ptr <!s32i>, !s32i
// CHECK-NEXT:   %5 = cir.const(#cir.int<1> : !s32i) : !s32i
// CHECK-NEXT:   %6 = cir.binop(add, %4, %5) : !s32i
// CHECK-NEXT:   cir.store %6, %0 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT:   cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT:   %4 = cir.load %2 : cir.ptr <!s32i>, !s32i
// CHECK-NEXT:   %5 = cir.const(#cir.int<1> : !s32i) : !s32i
// CHECK-NEXT:   %6 = cir.binop(add, %4, %5) : !s32i
// CHECK-NEXT:   cir.store %6, %2 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT:   cir.yield
// CHECK-NEXT: }

void l2(bool cond) {
  int i = 0;
  while (cond) {
    i = i + 1;
  }
  while (true) {
    i = i + 1;
  }
  while (1) {
    i = i + 1;
  }
}

// CHECK: cir.func @_Z2l2b
// CHECK:         cir.scope {
// CHECK-NEXT:     cir.while {
// CHECK-NEXT:       %3 = cir.load %0 : cir.ptr <!cir.bool>, !cir.bool
// CHECK-NEXT:       cir.condition(%3)
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %3 = cir.load %1 : cir.ptr <!s32i>, !s32i
// CHECK-NEXT:       %4 = cir.const(#cir.int<1> : !s32i) : !s32i
// CHECK-NEXT:       %5 = cir.binop(add, %3, %4) : !s32i
// CHECK-NEXT:       cir.store %5, %1 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT:       cir.yield
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   cir.scope {
// CHECK-NEXT:     cir.while {
// CHECK-NEXT:       %[[#TRUE:]] = cir.const(#true) : !cir.bool
// CHECK-NEXT:       cir.condition(%[[#TRUE]])
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %3 = cir.load %1 : cir.ptr <!s32i>, !s32i
// CHECK-NEXT:       %4 = cir.const(#cir.int<1> : !s32i) : !s32i
// CHECK-NEXT:       %5 = cir.binop(add, %3, %4) : !s32i
// CHECK-NEXT:       cir.store %5, %1 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT:       cir.yield
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   cir.scope {
// CHECK-NEXT:     cir.while {
// CHECK-NEXT:       %3 = cir.const(#cir.int<1> : !s32i) : !s32i
// CHECK-NEXT:       %4 = cir.cast(int_to_bool, %3 : !s32i), !cir.bool
// CHECK-NEXT:       cir.condition(%4)
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %3 = cir.load %1 : cir.ptr <!s32i>, !s32i
// CHECK-NEXT:       %4 = cir.const(#cir.int<1> : !s32i) : !s32i
// CHECK-NEXT:       %5 = cir.binop(add, %3, %4) : !s32i
// CHECK-NEXT:       cir.store %5, %1 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT:       cir.yield
// CHECK-NEXT:     }
// CHECK-NEXT:   }

void l3(bool cond) {
  int i = 0;
  do {
    i = i + 1;
  } while (cond);
  do {
    i = i + 1;
  } while (true);
  do {
    i = i + 1;
  } while (1);
}

// CHECK: cir.func @_Z2l3b
// CHECK: cir.scope {
// CHECK-NEXT:   cir.do {
// CHECK-NEXT:     %3 = cir.load %1 : cir.ptr <!s32i>, !s32i
// CHECK-NEXT:     %4 = cir.const(#cir.int<1> : !s32i) : !s32i
// CHECK-NEXT:     %5 = cir.binop(add, %3, %4) : !s32i
// CHECK-NEXT:     cir.store %5, %1 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT:     cir.yield
// CHECK-NEXT:   } while {
// CHECK-NEXT:     %[[#TRUE:]] = cir.load %0 : cir.ptr <!cir.bool>, !cir.bool
// CHECK-NEXT:     cir.condition(%[[#TRUE]])
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: cir.scope {
// CHECK-NEXT:   cir.do {
// CHECK-NEXT:     %3 = cir.load %1 : cir.ptr <!s32i>, !s32i
// CHECK-NEXT:     %4 = cir.const(#cir.int<1> : !s32i) : !s32i
// CHECK-NEXT:     %5 = cir.binop(add, %3, %4) : !s32i
// CHECK-NEXT:     cir.store %5, %1 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT:     cir.yield
// CHECK-NEXT:   } while {
// CHECK-NEXT:     %[[#TRUE:]] = cir.const(#true) : !cir.bool
// CHECK-NEXT:     cir.condition(%[[#TRUE]])
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: cir.scope {
// CHECK-NEXT:   cir.do {
// CHECK-NEXT:     %3 = cir.load %1 : cir.ptr <!s32i>, !s32i
// CHECK-NEXT:     %4 = cir.const(#cir.int<1> : !s32i) : !s32i
// CHECK-NEXT:     %5 = cir.binop(add, %3, %4) : !s32i
// CHECK-NEXT:     cir.store %5, %1 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT:     cir.yield
// CHECK-NEXT:   } while {
// CHECK-NEXT:     %3 = cir.const(#cir.int<1> : !s32i) : !s32i
// CHECK-NEXT:     %4 = cir.cast(int_to_bool, %3 : !s32i), !cir.bool
// CHECK-NEXT:     cir.condition(%4)
// CHECK-NEXT:   }
// CHECK-NEXT: }

void l4() {
  int i = 0, y = 100;
  while (true) {
    i = i + 1;
    if (i < 10)
      continue;
    y = y - 20;
  }
}

// CHECK: cir.func @_Z2l4v
// CHECK: cir.while {
// CHECK-NEXT:   %[[#TRUE:]] = cir.const(#true) : !cir.bool
// CHECK-NEXT:   cir.condition(%[[#TRUE]])
// CHECK-NEXT: } do {
// CHECK-NEXT:   %4 = cir.load %0 : cir.ptr <!s32i>, !s32i
// CHECK-NEXT:   %5 = cir.const(#cir.int<1> : !s32i) : !s32i
// CHECK-NEXT:   %6 = cir.binop(add, %4, %5) : !s32i
// CHECK-NEXT:   cir.store %6, %0 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT:   cir.scope {
// CHECK-NEXT:     %10 = cir.load %0 : cir.ptr <!s32i>, !s32i
// CHECK-NEXT:     %11 = cir.const(#cir.int<10> : !s32i) : !s32i
// CHECK-NEXT:     %12 = cir.cmp(lt, %10, %11) : !s32i, !cir.bool
// CHECK-NEXT:     cir.if %12 {
// CHECK-NEXT:       cir.continue
// CHECK-NEXT:     }
// CHECK-NEXT:   }

void l5() {
  do {
  } while (0);
}

// CHECK: cir.func @_Z2l5v()
// CHECK-NEXT:   cir.scope {
// CHECK-NEXT:     cir.do {
// CHECK-NEXT:       cir.yield
// CHECK-NEXT:     } while {
// CHECK-NEXT:       %0 = cir.const(#cir.int<0> : !s32i) : !s32i
// CHECK-NEXT:       %1 = cir.cast(int_to_bool, %0 : !s32i), !cir.bool
// CHECK-NEXT:       cir.condition(%1)
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }

void l6() {
  while (true) {
    return;
  }
}

// CHECK: cir.func @_Z2l6v()
// CHECK-NEXT:   cir.scope {
// CHECK-NEXT:     cir.while {
// CHECK-NEXT:       %[[#TRUE:]] = cir.const(#true) : !cir.bool
// CHECK-NEXT:       cir.condition(%[[#TRUE]])
// CHECK-NEXT:     } do {
// CHECK-NEXT:       cir.return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }
