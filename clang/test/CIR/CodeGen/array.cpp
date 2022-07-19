// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -Wno-return-stack-address -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// XFAIL: *

void a0() {
  int a[10];
}

// CHECK: func @_Z2a0v() {
// CHECK-NEXT:   %0 = cir.alloca !cir.array<i32 x 10>, cir.ptr <!cir.array<i32 x 10>>, ["a", uninitialized] {alignment = 16 : i64}

void a1() {
  int a[10];
  a[0] = 1;
}

// CHECK: func @_Z2a1v() {
// CHECK-NEXT:  %0 = cir.alloca !cir.array<i32 x 10>, cir.ptr <!cir.array<i32 x 10>>, ["a", uninitialized] {alignment = 16 : i64}
// CHECK-NEXT:  %1 = cir.cst(1 : i32) : i32
// CHECK-NEXT:  %2 = cir.cst(0 : i32) : i32
// CHECK-NEXT:  %3 = cir.cast(array_to_ptrdecay, %0 : !cir.ptr<!cir.array<i32 x 10>>), !cir.ptr<i32>
// CHECK-NEXT:  %4 = cir.ptr_stride(%3 : !cir.ptr<i32>, %2 : i32), !cir.ptr<i32>
// CHECK-NEXT:  cir.store %1, %4 : i32, cir.ptr <i32>

int *a2() {
  int a[4];
  return &a[0];
}

// CHECK: func @_Z2a2v() -> !cir.ptr<i32> {
// CHECK-NEXT:   %0 = cir.alloca !cir.ptr<i32>, cir.ptr <!cir.ptr<i32>>, ["__retval", uninitialized] {alignment = 8 : i64}
// CHECK-NEXT:   %1 = cir.alloca !cir.array<i32 x 4>, cir.ptr <!cir.array<i32 x 4>>, ["a", uninitialized] {alignment = 16 : i64}
// CHECK-NEXT:   %2 = cir.cst(0 : i32) : i32
// CHECK-NEXT:   %3 = cir.cast(array_to_ptrdecay, %1 : !cir.ptr<!cir.array<i32 x 4>>), !cir.ptr<i32>
// CHECK-NEXT:   %4 = cir.ptr_stride(%3 : !cir.ptr<i32>, %2 : i32), !cir.ptr<i32>
// CHECK-NEXT:   cir.store %4, %0 : !cir.ptr<i32>, cir.ptr <!cir.ptr<i32>>
// CHECK-NEXT:   %5 = cir.load %0 : cir.ptr <!cir.ptr<i32>>, !cir.ptr<i32>
// CHECK-NEXT:   cir.return %5 : !cir.ptr<i32>

void local_stringlit() {
  const char *s = "whatnow";
}

// CHECK: cir.global "private" constant internal @".str" = #cir.cst_array<"whatnow\00" : !cir.array<i8 x 8>> : !cir.array<i8 x 8> {alignment = 1 : i64} loc(#loc17)
// CHECK: func @_Z15local_stringlitv() {
// CHECK-NEXT:  %0 = cir.alloca !cir.ptr<i8>, cir.ptr <!cir.ptr<i8>>, ["s", cinit] {alignment = 8 : i64}
// CHECK-NEXT:  %1 = cir.get_global @".str" : cir.ptr <!cir.array<i8 x 8>>
// CHECK-NEXT:  %2 = cir.cast(array_to_ptrdecay, %1 : !cir.ptr<!cir.array<i8 x 8>>), !cir.ptr<i8>
// CHECK-NEXT:  cir.store %2, %0 : !cir.ptr<i8>, cir.ptr <!cir.ptr<i8>>
