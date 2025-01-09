// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck %s --check-prefix=CIR --input-file=%t.cir

auto func() {
  return __builtin_strcmp("", "");
  // CIR:      cir.func @_Z4funcv()
  // CIR-NEXT: %0 = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
  // CIR-NEXT: %1 = cir.const #cir.int<0> : !s32i
  // CIR-NEXT: cir.store %1, %0 : !s32i, !cir.ptr<!s32i>
  // CIR-NEXT: %2 = cir.load %0 : !cir.ptr<!s32i>, !s32i
  // CIR-NEXT: cir.return %2 : !s32i
}
