// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s
// RUN: %clang_cc1 -x c++ -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CXX

void a(void) {}
int b(int a, int b) {
  return a + b;
}
double c(double a, double b) {
  return a + b;
}

void d(void) {
  a();
  b(0, 1);
}

// CHECK: module {{.*}} {
// CHECK:   cir.func @a()
// CHECK:     cir.return
// CHECK:   }
// CHECK:   cir.func @b(%arg0: !s32i {{.*}}, %arg1: !s32i {{.*}}) -> !s32i
// CHECK:     %0 = cir.alloca !s32i, cir.ptr <!s32i>, ["a", init]
// CHECK:     %1 = cir.alloca !s32i, cir.ptr <!s32i>, ["b", init]
// CHECK:     %2 = cir.alloca !s32i, cir.ptr <!s32i>, ["__retval"]
// CHECK:     cir.store %arg0, %0 : !s32i, cir.ptr <!s32i>
// CHECK:     cir.store %arg1, %1 : !s32i, cir.ptr <!s32i>
// CHECK:     %3 = cir.load %0 : cir.ptr <!s32i>, !s32i
// CHECK:     %4 = cir.load %1 : cir.ptr <!s32i>, !s32i
// CHECK:     %5 = cir.binop(add, %3, %4) : !s32i
// CHECK:     cir.store %5, %2 : !s32i, cir.ptr <!s32i>
// CHECK:     %6 = cir.load %2 : cir.ptr <!s32i>, !s32i
// CHECK:     cir.return %6
// CHECK:   }
// CHECK:   cir.func @c(%arg0: !cir.double {{.*}}, %arg1: !cir.double {{.*}}) -> !cir.double
// CHECK:     %0 = cir.alloca !cir.double, cir.ptr <!cir.double>, ["a", init]
// CHECK:     %1 = cir.alloca !cir.double, cir.ptr <!cir.double>, ["b", init]
// CHECK:     %2 = cir.alloca !cir.double, cir.ptr <!cir.double>, ["__retval"]
// CHECK:     cir.store %arg0, %0 : !cir.double, cir.ptr <!cir.double>
// CHECK:     cir.store %arg1, %1 : !cir.double, cir.ptr <!cir.double>
// CHECK:     %3 = cir.load %0 : cir.ptr <!cir.double>, !cir.double
// CHECK:     %4 = cir.load %1 : cir.ptr <!cir.double>, !cir.double
// CHECK:     %5 = cir.binop(add, %3, %4) : !cir.double
// CHECK:     cir.store %5, %2 : !cir.double, cir.ptr <!cir.double>
// CHECK:     %6 = cir.load %2 : cir.ptr <!cir.double>, !cir.double
// CHECK:     cir.return %6 : !cir.double
// CHECK:   }
// CHECK:   cir.func @d()
// CHECK:     call @a() : () -> ()
// CHECK:     %0 = cir.const(#cir.int<0> : !s32i) : !s32i
// CHECK:     %1 = cir.const(#cir.int<1> : !s32i) : !s32i
// CHECK:     call @b(%0, %1) : (!s32i, !s32i) -> !s32i
// CHECK:     cir.return
// CHECK:   }
//
// CXX: module {{.*}} {
// CXX-NEXT:   cir.func @_Z1av()
// CXX-NEXT:     cir.return
// CXX-NEXT:   }
// CXX-NEXT:   cir.func @_Z1bii(%arg0: !s32i {{.*}}, %arg1: !s32i {{.*}}) -> !s32i
// CXX-NEXT:     %0 = cir.alloca !s32i, cir.ptr <!s32i>, ["a", init]
// CXX-NEXT:     %1 = cir.alloca !s32i, cir.ptr <!s32i>, ["b", init]
// CXX-NEXT:     %2 = cir.alloca !s32i, cir.ptr <!s32i>, ["__retval"]
// CXX-NEXT:     cir.store %arg0, %0 : !s32i, cir.ptr <!s32i>
// CXX-NEXT:     cir.store %arg1, %1 : !s32i, cir.ptr <!s32i>
// CXX-NEXT:     %3 = cir.load %0 : cir.ptr <!s32i>, !s32i
// CXX-NEXT:     %4 = cir.load %1 : cir.ptr <!s32i>, !s32i
// CXX-NEXT:     %5 = cir.binop(add, %3, %4) : !s32i
// CXX-NEXT:     cir.store %5, %2 : !s32i, cir.ptr <!s32i>
// CXX-NEXT:     %6 = cir.load %2 : cir.ptr <!s32i>, !s32i
// CXX-NEXT:     cir.return %6
// CXX-NEXT:   }
// CXX-NEXT:   cir.func @_Z1cdd(%arg0: !cir.double {{.*}}, %arg1: !cir.double {{.*}}) -> !cir.double
// CXX-NEXT:     %0 = cir.alloca !cir.double, cir.ptr <!cir.double>, ["a", init]
// CXX-NEXT:     %1 = cir.alloca !cir.double, cir.ptr <!cir.double>, ["b", init]
// CXX-NEXT:     %2 = cir.alloca !cir.double, cir.ptr <!cir.double>, ["__retval"]
// CXX-NEXT:     cir.store %arg0, %0 : !cir.double, cir.ptr <!cir.double>
// CXX-NEXT:     cir.store %arg1, %1 : !cir.double, cir.ptr <!cir.double>
// CXX-NEXT:     %3 = cir.load %0 : cir.ptr <!cir.double>, !cir.double
// CXX-NEXT:     %4 = cir.load %1 : cir.ptr <!cir.double>, !cir.double
// CXX-NEXT:     %5 = cir.binop(add, %3, %4) : !cir.double
// CXX-NEXT:     cir.store %5, %2 : !cir.double, cir.ptr <!cir.double>
// CXX-NEXT:     %6 = cir.load %2 : cir.ptr <!cir.double>, !cir.double
// CXX-NEXT:     cir.return %6 : !cir.double
// CXX-NEXT:   }
// CXX-NEXT:   cir.func @_Z1dv()
// CXX-NEXT:     call @_Z1av() : () -> ()
// CXX-NEXT:     %0 = cir.const(#cir.int<0> : !s32i) : !s32i
// CXX-NEXT:     %1 = cir.const(#cir.int<1> : !s32i) : !s32i
// CXX-NEXT:     call @_Z1bii(%0, %1) : (!s32i, !s32i) -> !s32i
// CXX-NEXT:     cir.return
// CXX-NEXT:   }
