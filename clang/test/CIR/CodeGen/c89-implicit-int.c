// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c89 -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// Implicit int return type.
test = 0;
// CHECK: cir.global external @test = #cir.int<0> : !s32i
func (void) {
// CHECK: cir.func dso_local @func() -> !s32i
  return 0;
}
