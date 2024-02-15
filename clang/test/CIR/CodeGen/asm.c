// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

//CHECK: cir.asm(x86_att, {"" ""})  : () -> ()
void empty1() {
  __asm__ volatile("" : : : );
}

//CHECK: cir.asm(x86_att, {"xyz" ""})  : () -> () 
void empty2() {
  __asm__ volatile("xyz" : : : );
}

//CHECK: cir.asm(x86_att, {"" "=*m,*m"}) %0, %0 : (!cir.ptr<!s32i>, !cir.ptr<!s32i>) -> ()
void t1(int x) {
  __asm__ volatile("" : "+m"(x));
}

//CHECK: cir.asm(x86_att, {"" "*m"}) %0 : (!cir.ptr<!s32i>) -> ()
void t2(int x) {
  __asm__ volatile("" : : "m"(x));
}

//CHECK: cir.asm(x86_att, {"" "=*m"}) %0 : (!cir.ptr<!s32i>) -> ()
void t3(int x) {
  __asm__ volatile("" : "=m"(x));
}

//CHECK: cir.asm(x86_att, {"" "=&r,=&r,1"}) %1 : (!s32i) -> ()
void t4(int x) {
  __asm__ volatile("" : "=&r"(x), "+&r"(x));
}