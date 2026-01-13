// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM

void f1() {
  // CIR: cir.asm(x86_att, 
  // CIR:   out = [],
  // CIR:   in = [],
  // CIR:   in_out = [],
  // CIR:   {"" "~{dirflag},~{fpsr},~{flags}"}) side_effects
  // LLVM: call void asm sideeffect "", "~{dirflag},~{fpsr},~{flags}"()
  __asm__ volatile("" : : : );
}

void f2() {
  // CIR: cir.asm(x86_att,
  // CIR:   out = [],
  // CIR:   in = [],
  // CIR:   in_out = [],
  // CIR:   {"nop" "~{dirflag},~{fpsr},~{flags}"}) side_effects
  // LLVM: call void asm sideeffect "nop", "~{dirflag},~{fpsr},~{flags}"()
  __asm__ volatile("nop" : : : );
}
