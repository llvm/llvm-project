// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

void if0() {
  int x = 0;
  if constexpr (0 == 0) {
    // Declare a variable with same name to be sure we handle the
    // scopes correctly
    int x = 2;
  } else {
    int x = 3;
  }
  if constexpr (0 == 1) {
    int x = 4;
  } else {
    int x = 5;
  }
  if constexpr (int x = 7; 8 == 8) {
    int y = x;
  } else {
    int y = 2*x;
  }
  if constexpr (int x = 9; 8 == 10) {
    int y = x;
  } else {
    int y = 3*x;
  }
  if constexpr (10 == 10) {
    int x = 20;
  }
  if constexpr (10 == 11) {
    int x = 30;
  }
  if constexpr (int x = 70; 80 == 80) {
    int y = 10*x;
  }
  if constexpr (int x = 90; 80 == 100) {
    int y = 11*x;
  }
}

// CHECK: cir.func @_Z3if0v() {{.*}}
// CHECK: cir.store %1, %0 : !s32i, cir.ptr <!s32i> loc({{.*}})
// CHECK-NEXT: cir.scope {
// CHECK-NEXT:   %2 = cir.alloca !s32i, cir.ptr <!s32i>, ["x", init] {{.*}}
// CHECK-NEXT:   %3 = cir.const(#cir.int<2> : !s32i) : !s32i loc({{.*}})
// CHECK-NEXT:   cir.store %3, %2 : !s32i, cir.ptr <!s32i> loc({{.*}})
// CHECK-NEXT: } loc({{.*}})
// CHECK-NEXT: cir.scope {
// CHECK-NEXT:   %2 = cir.alloca !s32i, cir.ptr <!s32i>, ["x", init] {{.*}}
// CHECK-NEXT:   %3 = cir.const(#cir.int<5> : !s32i) : !s32i loc({{.*}})
// CHECK-NEXT:   cir.store %3, %2 : !s32i, cir.ptr <!s32i> loc({{.*}})
// CHECK-NEXT: } loc({{.*}})
// CHECK-NEXT: cir.scope {
// CHECK-NEXT:   %2 = cir.alloca !s32i, cir.ptr <!s32i>, ["x", init] {{.*}}
// CHECK-NEXT:   %3 = cir.alloca !s32i, cir.ptr <!s32i>, ["y", init] {{.*}}
// CHECK-NEXT:   %4 = cir.const(#cir.int<7> : !s32i) : !s32i loc({{.*}})
// CHECK-NEXT:   cir.store %4, %2 : !s32i, cir.ptr <!s32i> loc({{.*}})
// CHECK-NEXT:   %5 = cir.load %2 : cir.ptr <!s32i>, !s32i loc({{.*}})
// CHECK-NEXT:   cir.store %5, %3 : !s32i, cir.ptr <!s32i> loc({{.*}})
// CHECK-NEXT: } loc({{.*}})
// CHECK-NEXT: cir.scope {
// CHECK-NEXT:   %2 = cir.alloca !s32i, cir.ptr <!s32i>, ["x", init] {{.*}}
// CHECK-NEXT:   %3 = cir.alloca !s32i, cir.ptr <!s32i>, ["y", init] {{.*}}
// CHECK-NEXT:   %4 = cir.const(#cir.int<9> : !s32i) : !s32i loc({{.*}})
// CHECK-NEXT:   cir.store %4, %2 : !s32i, cir.ptr <!s32i> loc({{.*}})
// CHECK-NEXT:   %5 = cir.const(#cir.int<3> : !s32i) : !s32i loc({{.*}})
// CHECK-NEXT:   %6 = cir.load %2 : cir.ptr <!s32i>, !s32i loc({{.*}})
// CHECK-NEXT:   %7 = cir.binop(mul, %5, %6) : !s32i loc({{.*}})
// CHECK-NEXT:   cir.store %7, %3 : !s32i, cir.ptr <!s32i> loc({{.*}})
// CHECK-NEXT: } loc({{.*}})
// CHECK-NEXT: cir.scope {
// CHECK-NEXT:   %2 = cir.alloca !s32i, cir.ptr <!s32i>, ["x", init] {{.*}}
// CHECK-NEXT:   %3 = cir.const(#cir.int<20> : !s32i) : !s32i loc({{.*}})
// CHECK-NEXT:   cir.store %3, %2 : !s32i, cir.ptr <!s32i> loc({{.*}})
// CHECK-NEXT: } loc({{.*}})
// CHECK-NEXT: cir.scope {
// Note that Clang does not even emit a block in this case
// CHECK-NEXT: } loc({{.*}})
// CHECK-NEXT: cir.scope {
// CHECK-NEXT:   %2 = cir.alloca !s32i, cir.ptr <!s32i>, ["x", init] {{.*}}
// CHECK-NEXT:   %3 = cir.alloca !s32i, cir.ptr <!s32i>, ["y", init] {{.*}}
// CHECK-NEXT:   %4 = cir.const(#cir.int<70> : !s32i) : !s32i loc({{.*}})
// CHECK-NEXT:   cir.store %4, %2 : !s32i, cir.ptr <!s32i> loc({{.*}})
// CHECK-NEXT:   %5 = cir.const(#cir.int<10> : !s32i) : !s32i loc({{.*}})
// CHECK-NEXT:   %6 = cir.load %2 : cir.ptr <!s32i>, !s32i loc({{.*}})
// CHECK-NEXT:   %7 = cir.binop(mul, %5, %6) : !s32i loc({{.*}})
// CHECK-NEXT:   cir.store %7, %3 : !s32i, cir.ptr <!s32i> loc({{.*}})
// CHECK-NEXT: } loc({{.*}})
// CHECK-NEXT: cir.scope {
// CHECK-NEXT:   %2 = cir.alloca !s32i, cir.ptr <!s32i>, ["x", init] {{.*}}
// CHECK-NEXT:   %3 = cir.const(#cir.int<90> : !s32i) : !s32i loc({{.*}})
// CHECK-NEXT:   cir.store %3, %2 : !s32i, cir.ptr <!s32i> loc({{.*}})
// CHECK-NEXT: } loc({{.*}})
// CHECK-NEXT: cir.return loc({{.*}})
