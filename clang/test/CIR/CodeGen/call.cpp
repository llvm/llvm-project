// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

int& p();
int f() {
  return p() - 22;
}

// CHECK: cir.func @_Z1fv() -> !s32i {
// CHECK:   %0 = cir.alloca !s32i, cir.ptr <!s32i>, ["__retval"] {alignment = 4 : i64}
// CHECK:   %1 = cir.call @_Z1pv() : () -> !cir.ptr<!s32i>
// CHECK:   %2 = cir.load %1 : cir.ptr <!s32i>, !s32i
// CHECK:   %3 = cir.const(#cir.int<22> : !s32i) : !s32i
// CHECK:   %4 = cir.binop(sub, %2, %3) : !s32i