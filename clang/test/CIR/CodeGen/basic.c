// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

int foo(int i);

int foo(int i) {
  i;
  return i;
}

//      CHECK: module attributes {
// CHECK-NEXT: cir.func @foo(%arg0: !s32i loc({{.*}})) -> !s32i {
// CHECK-NEXT: %0 = cir.alloca !s32i, cir.ptr <!s32i>, ["i", init] {alignment = 4 : i64}
// CHECK-NEXT: %1 = cir.alloca !s32i, cir.ptr <!s32i>, ["__retval"] {alignment = 4 : i64}
// CHECK-NEXT: cir.store %arg0, %0 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT: %2 = cir.load %0 : cir.ptr <!s32i>, !s32i
// CHECK-NEXT: %3 = cir.load %0 : cir.ptr <!s32i>, !s32i
// CHECK-NEXT: cir.store %3, %1 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT: %4 = cir.load %1 : cir.ptr <!s32i>, !s32i
// CHECK-NEXT: cir.return %4 : !s32i

int f2() { return 3; }

// CHECK: cir.func @f2() -> !s32i {
// CHECK-NEXT: %0 = cir.alloca !s32i, cir.ptr <!s32i>, ["__retval"] {alignment = 4 : i64}
// CHECK-NEXT: %1 = cir.const(#cir.int<3> : !s32i) : !s32i
// CHECK-NEXT: cir.store %1, %0 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT: %2 = cir.load %0 : cir.ptr <!s32i>, !s32i
// CHECK-NEXT: cir.return %2 : !s32i

int f3() {
  int i = 3;
  return i;
}

// CHECK: cir.func @f3() -> !s32i {
// CHECK-NEXT: %0 = cir.alloca !s32i, cir.ptr <!s32i>, ["__retval"] {alignment = 4 : i64}
// CHECK-NEXT: %1 = cir.alloca !s32i, cir.ptr <!s32i>, ["i", init] {alignment = 4 : i64}
// CHECK-NEXT: %2 = cir.const(#cir.int<3> : !s32i) : !s32i
// CHECK-NEXT: cir.store %2, %1 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT: %3 = cir.load %1 : cir.ptr <!s32i>, !s32i
// CHECK-NEXT: cir.store %3, %0 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT: %4 = cir.load %0 : cir.ptr <!s32i>, !s32i
// CHECK-NEXT: cir.return %4 : !s32i
