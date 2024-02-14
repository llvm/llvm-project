// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s

auto func() {
  return __builtin_strcmp("", "");
  // CHECK:      cir.func @_Z4funcv()
  // CHECK-NEXT: %0 = cir.alloca !s32i, cir.ptr <!s32i>, ["__retval"] {alignment = 4 : i64} loc(#loc2)
  // CHECK-NEXT: %1 = cir.const(#cir.int<0> : !s32i) : !s32i loc(#loc7)
  // CHECK-NEXT: cir.store %1, %0 : !s32i, cir.ptr <!s32i> loc(#loc8)
  // CHECK-NEXT: %2 = cir.load %0 : cir.ptr <!s32i>, !s32i loc(#loc8)
  // CHECK-NEXT: cir.return %2 : !s32i loc(#loc8)
}
