// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// XFAIL: *

void l0() {
  for (;;) {
  }
}

// CHECK: func @l0
// CHECK: cir.loop for(cond :  {
// CHECK-NEXT:   %0 = cir.cst(true) : !cir.bool
// CHECK-NEXT:   cir.yield loopcondition %0 : !cir.bool
// CHECK-NEXT: }, step :  {
// CHECK-NEXT:   cir.yield
// CHECK-NEXT: })  {
// CHECK-NEXT:   cir.yield
// CHECK-NEXT: }

void l1() {
  int x = 0;
  for (int i = 0; i < 10; i = i + 1) {
    x = x + 1;
  }
}

// CHECK: func @l1
// CHECK: cir.loop for(cond :  {
// CHECK-NEXT:   %4 = cir.load %2 : cir.ptr <i32>, i32
// CHECK-NEXT:   %5 = cir.cst(10 : i32) : i32
// CHECK-NEXT:   %6 = cir.cmp(lt, %4, %5) : i32, !cir.bool
// CHECK-NEXT:   cir.yield loopcondition %6 : !cir.bool
// CHECK-NEXT: }, step :  {
// CHECK-NEXT:   %4 = cir.load %2 : cir.ptr <i32>, i32
// CHECK-NEXT:   %5 = cir.cst(1 : i32) : i32
// CHECK-NEXT:   %6 = cir.binop(add, %4, %5) : i32
// CHECK-NEXT:   cir.store %6, %2 : i32, cir.ptr <i32>
// CHECK-NEXT:   cir.yield
// CHECK-NEXT: })  {
// CHECK-NEXT:   %4 = cir.load %0 : cir.ptr <i32>, i32
// CHECK-NEXT:   %5 = cir.cst(1 : i32) : i32
// CHECK-NEXT:   %6 = cir.binop(add, %4, %5) : i32
// CHECK-NEXT:   cir.store %6, %0 : i32, cir.ptr <i32>
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

// CHECK: func @l2
// CHECK:         cir.scope {
// CHECK-NEXT:     cir.loop while(cond :  {
// CHECK-NEXT:       %3 = cir.load %0 : cir.ptr <!cir.bool>, !cir.bool
// CHECK-NEXT:       cir.yield loopcondition %3 : !cir.bool
// CHECK-NEXT:     }, step :  {
// CHECK-NEXT:       cir.yield
// CHECK-NEXT:     })  {
// CHECK-NEXT:       %3 = cir.load %1 : cir.ptr <i32>, i32
// CHECK-NEXT:       %4 = cir.cst(1 : i32) : i32
// CHECK-NEXT:       %5 = cir.binop(add, %3, %4) : i32
// CHECK-NEXT:       cir.store %5, %1 : i32, cir.ptr <i32>
// CHECK-NEXT:       cir.yield
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   cir.scope {
// CHECK-NEXT:     cir.loop while(cond :  {
// CHECK-NEXT:       %3 = cir.cst(true) : !cir.bool
// CHECK-NEXT:       cir.yield loopcondition %3 : !cir.bool
// CHECK-NEXT:     }, step :  {
// CHECK-NEXT:       cir.yield
// CHECK-NEXT:     })  {
// CHECK-NEXT:       %3 = cir.load %1 : cir.ptr <i32>, i32
// CHECK-NEXT:       %4 = cir.cst(1 : i32) : i32
// CHECK-NEXT:       %5 = cir.binop(add, %3, %4) : i32
// CHECK-NEXT:       cir.store %5, %1 : i32, cir.ptr <i32>
// CHECK-NEXT:       cir.yield
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   cir.scope {
// CHECK-NEXT:     cir.loop while(cond :  {
// CHECK-NEXT:       %3 = cir.cst(1 : i32) : i32
// CHECK-NEXT:       %4 = cir.cast(int_to_bool, %3 : i32), !cir.bool
// CHECK-NEXT:       cir.yield loopcondition %4 : !cir.bool
// CHECK-NEXT:     }, step :  {
// CHECK-NEXT:       cir.yield
// CHECK-NEXT:     })  {
// CHECK-NEXT:       %3 = cir.load %1 : cir.ptr <i32>, i32
// CHECK-NEXT:       %4 = cir.cst(1 : i32) : i32
// CHECK-NEXT:       %5 = cir.binop(add, %3, %4) : i32
// CHECK-NEXT:       cir.store %5, %1 : i32, cir.ptr <i32>
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

// CHECK: func @l3
// CHECK: cir.scope {
// CHECK-NEXT:   cir.loop dowhile(cond :  {
// CHECK-NEXT:   %3 = cir.load %0 : cir.ptr <!cir.bool>, !cir.bool
// CHECK-NEXT:   cir.yield loopcondition %3 : !cir.bool
// CHECK-NEXT:   }, step :  {
// CHECK-NEXT:   cir.yield
// CHECK-NEXT:   })  {
// CHECK-NEXT:   %3 = cir.load %1 : cir.ptr <i32>, i32
// CHECK-NEXT:   %4 = cir.cst(1 : i32) : i32
// CHECK-NEXT:   %5 = cir.binop(add, %3, %4) : i32
// CHECK-NEXT:   cir.store %5, %1 : i32, cir.ptr <i32>
// CHECK-NEXT:   cir.yield
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: cir.scope {
// CHECK-NEXT:   cir.loop dowhile(cond :  {
// CHECK-NEXT:   %3 = cir.cst(true) : !cir.bool
// CHECK-NEXT:   cir.yield loopcondition %3 : !cir.bool
// CHECK-NEXT:   }, step :  {
// CHECK-NEXT:   cir.yield
// CHECK-NEXT:   })  {
// CHECK-NEXT:   %3 = cir.load %1 : cir.ptr <i32>, i32
// CHECK-NEXT:   %4 = cir.cst(1 : i32) : i32
// CHECK-NEXT:   %5 = cir.binop(add, %3, %4) : i32
// CHECK-NEXT:   cir.store %5, %1 : i32, cir.ptr <i32>
// CHECK-NEXT:   cir.yield
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: cir.scope {
// CHECK-NEXT:   cir.loop dowhile(cond :  {
// CHECK-NEXT:   %3 = cir.cst(1 : i32) : i32
// CHECK-NEXT:   %4 = cir.cast(int_to_bool, %3 : i32), !cir.bool
// CHECK-NEXT:   cir.yield loopcondition %4 : !cir.bool
// CHECK-NEXT:   }, step :  {
// CHECK-NEXT:   cir.yield
// CHECK-NEXT:   })  {
// CHECK-NEXT:   %3 = cir.load %1 : cir.ptr <i32>, i32
// CHECK-NEXT:   %4 = cir.cst(1 : i32) : i32
// CHECK-NEXT:   %5 = cir.binop(add, %3, %4) : i32
// CHECK-NEXT:   cir.store %5, %1 : i32, cir.ptr <i32>
// CHECK-NEXT:   cir.yield
// CHECK-NEXT:   }
// CHECK-NEXT: }
