// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
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

// CIR: %[[X:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init] {alignment = 4 : i64}
// CIR: cir.asm(x86_att, 
// CIR:   out = [%[[X]] : !cir.ptr<!s32i> (maybe_memory)],
// CIR:   in = [],
// CIR:   in_out = [],
// CIR:   {"" "=*m,~{dirflag},~{fpsr},~{flags}"}) side_effects
void f3(int x) {
  __asm__ volatile("" : "=m"(x));
}

// CIR: %[[A:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a"] {alignment = 4 : i64}
// CIR: %[[RES:.*]] = cir.asm(x86_att, 
// CIR:   out = [],
// CIR:   in = [],
// CIR:   in_out = [],
// CIR:   {"movl $$42, $0" "=r,~{dirflag},~{fpsr},~{flags}"}) -> !s32i
// CIR: cir.store align(4) %[[RES]], %[[A]] : !s32i, !cir.ptr<!s32i>
unsigned f4(unsigned x) {
  int a;
  __asm__("movl $42, %0" : "=r" (a) : );
  return a;
}

// CIR: [[TMP0:%.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["x", init] 
// CIR: cir.store{{.*}} %arg0, [[TMP0]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR: [[TMP1:%.*]] = cir.load deref{{.*}}  [[TMP0]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR: cir.asm(x86_att, 
// CIR:       out = [%1 : !cir.ptr<!s32i> (maybe_memory)],
// CIR:       in = [],
// CIR:       in_out = [],
// CIR:       {"addl $$42, $0" "=*m,~{dirflag},~{fpsr},~{flags}"}) 
// CIR-NEXT: cir.return
void f5(int *x) {    
  __asm__("addl $42, %[addr]" : [addr] "=m" (*x));
}
