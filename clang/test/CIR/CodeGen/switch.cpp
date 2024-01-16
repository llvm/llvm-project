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
// CHECK: cir.switch (%3 : !s32i) [
// CHECK-NEXT: case (equal, 0)  {
// CHECK-NEXT:   %4 = cir.load %1 : cir.ptr <!s32i>, !s32i
// CHECK-NEXT:   %5 = cir.const(#cir.int<1> : !s32i) : !s32i
// CHECK-NEXT:   %6 = cir.binop(add, %4, %5) : !s32i
// CHECK-NEXT:   cir.store %6, %1 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT:   cir.break
// CHECK-NEXT: },
// CHECK-NEXT: case (equal, 1)  {
// CHECK-NEXT:   cir.break
// CHECK-NEXT: },
// CHECK-NEXT: case (equal, 2)  {
// CHECK-NEXT:   cir.scope {
// CHECK-NEXT:       %4 = cir.alloca !s32i, cir.ptr <!s32i>, ["yolo", init]
// CHECK-NEXT:       %5 = cir.load %1 : cir.ptr <!s32i>, !s32i
// CHECK-NEXT:       %6 = cir.const(#cir.int<1> : !s32i) : !s32i
// CHECK-NEXT:       %7 = cir.binop(add, %5, %6) : !s32i
// CHECK-NEXT:       cir.store %7, %1 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT:       %8 = cir.const(#cir.int<100> : !s32i) : !s32i
// CHECK-NEXT:       cir.store %8, %4 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT:       cir.break
// CHECK-NEXT:     }
// CHECK-NEXT:     cir.yield
// CHECK-NEXT:   }

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
// CHECK-NEXT:   %1 = cir.alloca !s32i, cir.ptr <!s32i>, ["yolo", init]
// CHECK-NEXT:   %2 = cir.alloca !s32i, cir.ptr <!s32i>, ["fomo", init]
// CHECK:        cir.switch (%4 : !s32i) [
// CHECK-NEXT:   case (equal, 3)  {
// CHECK-NEXT:     %5 = cir.const(#cir.int<0> : !s32i) : !s32i
// CHECK-NEXT:     cir.store %5, %2 : !s32i, cir.ptr <!s32i>

void sw3(int a) {
  switch (a) {
  default:
    break;
  }
}

// CHECK: cir.func @_Z3sw3i
// CHECK: cir.scope {
// CHECK-NEXT:   %1 = cir.load %0 : cir.ptr <!s32i>, !s32i
// CHECK-NEXT:   cir.switch (%1 : !s32i) [
// CHECK-NEXT:   case (default)  {
// CHECK-NEXT:     cir.break
// CHECK-NEXT:   }
// CHECK-NEXT:   ]

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
// CHECK:       cir.switch (%4 : !s32i) [
// CHECK-NEXT:       case (equal, 42)  {
// CHECK-NEXT:         cir.scope {
// CHECK-NEXT:           %5 = cir.const(#cir.int<3> : !s32i) : !s32i
// CHECK-NEXT:           cir.store %5, %1 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT:           %6 = cir.load %1 : cir.ptr <!s32i>, !s32i
// CHECK-NEXT:           cir.return %6 : !s32i
// CHECK-NEXT:         }
// CHECK-NEXT:         cir.yield
// CHECK-NEXT:       },
// CHECK-NEXT:       case (default)  {
// CHECK-NEXT:         %5 = cir.const(#cir.int<2> : !s32i) : !s32i
// CHECK-NEXT:         cir.store %5, %1 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT:         %6 = cir.load %1 : cir.ptr <!s32i>, !s32i
// CHECK-NEXT:         cir.return %6 : !s32i
// CHECK-NEXT:       }
// CHECK-NEXT:       ]

void sw5(int a) {
  switch (a) {
  case 1:;
  }
}

// CHECK: cir.func @_Z3sw5i
// CHECK: cir.switch (%1 : !s32i) [
// CHECK-NEXT:   case (equal, 1)  {
// CHECK-NEXT:     cir.yield

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
// CHECK: cir.switch (%1 : !s32i) [
// CHECK-NEXT: case (anyof, [0, 1, 2] : !s32i)  {
// CHECK-NEXT:   cir.break
// CHECK-NEXT: },
// CHECK-NEXT: case (anyof, [3, 4, 5] : !s32i)  {
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
// CHECK: case (anyof, [0, 1, 2] : !s32i)  {
// CHECK-NEXT:   cir.yield
// CHECK-NEXT: },
// CHECK-NEXT: case (anyof, [3, 4, 5] : !s32i)  {
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
//CHECK:      case (equal, 3)
//CHECK-NEXT:   cir.break
//CHECK-NEXT: },
//CHECK-NEXT: case (equal, 4) {
//CHECK-NEXT:   cir.yield
//CHECK-NEXT: }
//CHECK-NEXT: case (default) {
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
//CHECK:      case (equal, 3) {
//CHECK-NEXT:   cir.break
//CHECK-NEXT: }
//CHECK-NEXT: case (default) {
//CHECK-NEXT:   cir.yield
//CHECK-NEXT: }
//CHECK:      case (equal, 4)
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
//CHECK:      case (equal, 3)
//CHECK-NEXT:   cir.break
//CHECK-NEXT: },
//CHECK-NEXT: case (equal, 4) {
//CHECK-NEXT:   cir.yield
//CHECK-NEXT: }
//CHECK-NEXT: case (default) {
//CHECK-NEXT:   cir.yield
//CHECK-NEXT: }
//CHECK-NEXT: case (equal, 5) {
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
//CHECK:      case (equal, 3)
//CHECK-NEXT:   cir.break
//CHECK-NEXT: },
//CHECK-NEXT: case (anyof, [4, 5] : !s32i) {
//CHECK-NEXT:   cir.yield
//CHECK-NEXT: }
//CHECK-NEXT: case (default) {
//CHECK-NEXT:   cir.yield
//CHECK-NEXT: }
//CHECK-NEXT: case (anyof, [6, 7] : !s32i)  {
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
// CHECK-NEXT:     case (equal, 3) {
// CHECK-NEXT:       cir.return
// CHECK-NEXT:     ^bb1:  // no predecessors
// CHECK-NEXT:       cir.break
// CHECK-NEXT:     }
