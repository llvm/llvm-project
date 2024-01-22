// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cpp.cir
// RUN: FileCheck --input-file=%t.cpp.cir %s --check-prefix=CPPSCOPE
// RUN: %clang_cc1 -x c -std=c11 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.c.cir
// RUN: FileCheck --input-file=%t.c.cir %s --check-prefix=CSCOPE

void l0(void) {
  for (int i = 0;;) {
    int j = 0;
  }
}

// CPPSCOPE: cir.func @_Z2l0v()
// CPPSCOPE-NEXT:   cir.scope {
// CPPSCOPE-NEXT:     %0 = cir.alloca !s32i, cir.ptr <!s32i>, ["i", init] {alignment = 4 : i64}
// CPPSCOPE-NEXT:     %1 = cir.alloca !s32i, cir.ptr <!s32i>, ["j", init] {alignment = 4 : i64}
// CPPSCOPE-NEXT:     %2 = cir.const(#cir.int<0> : !s32i) : !s32i
// CPPSCOPE-NEXT:     cir.store %2, %0 : !s32i, cir.ptr <!s32i>
// CPPSCOPE-NEXT:     cir.for : cond {

// CSCOPE: cir.func @l0()
// CSCOPE-NEXT: cir.scope {
// CSCOPE-NEXT:   %0 = cir.alloca !s32i, cir.ptr <!s32i>, ["i", init] {alignment = 4 : i64}
// CSCOPE-NEXT:   %1 = cir.const(#cir.int<0> : !s32i) : !s32i
// CSCOPE-NEXT:   cir.store %1, %0 : !s32i, cir.ptr <!s32i>
// CSCOPE-NEXT:   cir.for : cond {

// CSCOPE:        } body {
// CSCOPE-NEXT:     cir.scope {
// CSCOPE-NEXT:       %2 = cir.alloca !s32i, cir.ptr <!s32i>, ["j", init] {alignment = 4 : i64}
