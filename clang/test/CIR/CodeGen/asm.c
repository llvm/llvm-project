// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

//CHECK: cir.asm(x86_att, {""})
void empty1() {
  __asm__ volatile("" : : : );
}

//CHECK: cir.asm(x86_att, {"xyz"})
void empty2() {
  __asm__ volatile("xyz" : : : );
}