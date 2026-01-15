// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM

__asm__ ("foo1");
__asm__ ("foo2");
__asm__ ("foo3");
// CIR: module{{.*}} cir.module_asm = ["foo1", "foo2", "foo3"]
// LLVM: module asm "foo1"
// LLVM-NEXT: module asm "foo2"
// LLVM-NEXT: module asm "foo3"

//      CIR: cir.func{{.*}}@empty1
//      CIR: cir.asm(x86_att, 
// CIR-NEXT:   out = [],
// CIR-NEXT:   in = [],
// CIR-NEXT:   in_out = [],
// CIR-NEXT:   {"" "~{dirflag},~{fpsr},~{flags}"}) side_effects
// LLVM: define{{.*}}@empty1
// LLVM: call void asm sideeffect "", "~{dirflag},~{fpsr},~{flags}"()
void empty1() {
  __asm__ volatile("" : : : );
}

//      CIR: cir.func{{.*}}@empty2
//      CIR: cir.asm(x86_att,
// CIR-NEXT:   out = [],
// CIR-NEXT:   in = [],
// CIR-NEXT:   in_out = [],
// CIR-NEXT:   {"nop" "~{dirflag},~{fpsr},~{flags}"}) side_effects
// LLVM: define{{.*}}@empty2
// LLVM: call void asm sideeffect "nop", "~{dirflag},~{fpsr},~{flags}"()
void empty2() {
  __asm__ volatile("nop" : : : );
}

//      CIR: cir.func{{.*}}@empty5
//      CIR: %[[X:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]
//      CIR: cir.asm(x86_att, 
// CIR-NEXT:   out = [%[[X]] : !cir.ptr<!s32i> (maybe_memory)],
// CIR-NEXT:   in = [],
// CIR-NEXT:   in_out = [],
// CIR-NEXT:   {"" "=*m,~{dirflag},~{fpsr},~{flags}"}) side_effects
// LLVM: define{{.*}}@empty5
// LLVM: %[[X:.*]] = alloca i32
// LLVM: call void asm sideeffect "", "=*m,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i32) %[[X]])
void empty5(int x) {
  __asm__ volatile("" : "=m"(x));
}

//      CIR: cir.func{{.*}}@add4
//      CIR: %[[X:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["x", init] 
//      CIR: %[[X_LOAD:.*]] = cir.load {{.*}}  %[[X]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR-NEXT: cir.asm(x86_att, 
// CIR-NEXT:       out = [%[[X_LOAD]] : !cir.ptr<!s32i> (maybe_memory)],
// CIR-NEXT:       in = [],
// CIR-NEXT:       in_out = [],
// CIR-NEXT:       {"addl $$42, $0" "=*m,~{dirflag},~{fpsr},~{flags}"})
// CIR-NEXT: cir.return
// LLVM: define {{.*}}add4
// LLVM: %[[X:.*]] = alloca ptr
// LLVM: %[[X_LOAD:.*]] = load ptr, ptr %[[X]]
// LLVM: call void asm "addl $$42, $0", "=*m,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i32) %[[X_LOAD]])
void add4(int *x) {
  __asm__("addl $42, %[addr]" : [addr] "=m" (*x));
}

//      CIR: cir.func{{.*}}@mov
//      CIR: %[[A:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a"] {alignment = 4 : i64}
//      CIR: %[[RES:.*]] = cir.asm(x86_att, 
// CIR-NEXT:   out = [],
// CIR-NEXT:   in = [],
// CIR-NEXT:   in_out = [],
// CIR-NEXT:   {"movl $$42, $0" "=r,~{dirflag},~{fpsr},~{flags}"}) -> !s32i
// CIR: cir.store align(4) %[[RES]], %[[A]] : !s32i, !cir.ptr<!s32i>
// LLVM: define{{.*}}@mov
// LLVM: call i32 asm "movl $$42, $0", "=r,~{dirflag},~{fpsr},~{flags}"()
unsigned mov(unsigned x) {
  int a;
  __asm__("movl $42, %0" : "=r" (a) : );
  return a;
}

// bitfield destination of an asm.
struct S {
  int a : 4;
};

//      CIR: cir.func{{.*}}@t14
//      CIR: cir.asm(x86_att,
// CIR-NEXT:         out = [],
// CIR-NEXT:         in = [],
// CIR-NEXT:         in_out = [],
// CIR-NEXT:         {"abc $0" "=r,~{dirflag},~{fpsr},~{flags}"}) -> !s32i
// LLVM: define{{.*}}@t14
// LLVM: call i32 asm "abc $0", "=r,~{dirflag},~{fpsr},~{flags}"()
void t14(struct S *P) {
  __asm__("abc %0" : "=r"(P->a) );
}

struct large {
  int x[1000];
};

//      CIR: cir.func{{.*}}@t17
//      CIR: %[[I:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i"]
//      CIR: cir.asm(x86_att,
// CIR-NEXT:         out = [%[[I]] : !cir.ptr<!s32i> (maybe_memory)],
// CIR-NEXT:         in = [],
// CIR-NEXT:         in_out = [],
// CIR-NEXT:         {"nop" "=*m,~{dirflag},~{fpsr},~{flags}"})
// LLVM: define{{.*}}@t17
// LLVM: %[[I:.*]] = alloca i32
// LLVM: call void asm "nop", "=*m,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i32) %[[I]])
void t17(void) {
  int i;
  __asm__ ( "nop": "=m"(i));
}

//      CIR: cir.func{{.*}}@t25
//      CIR: cir.asm(x86_att,
// CIR-NEXT:         out = [],
// CIR-NEXT:         in = [],
// CIR-NEXT:         in_out = [],
// CIR-NEXT:         {"finit" "~{st},~{st(1)},~{st(2)},~{st(3)},~{st(4)},~{st(5)},~{st(6)},~{st(7)},~{fpsr},~{fpcr},~{dirflag},~{fpsr},~{flags}"}) side_effects
// LLVM: define{{.*}}@t25
// LLVM: call void asm sideeffect "finit", "~{st},~{st(1)},~{st(2)},~{st(3)},~{st(4)},~{st(5)},~{st(6)},~{st(7)},~{fpsr},~{fpcr},~{dirflag},~{fpsr},~{flags}"()
void t25(void)
{
  __asm__ __volatile__(					   \
		       "finit"				   \
		       :				   \
		       :				   \
		       :"st","st(1)","st(2)","st(3)",	   \
			"st(4)","st(5)","st(6)","st(7)",   \
			"fpsr","fpcr"			   \
							   );
}

//t26 skipped - no vector type support

//      CIR: cir.func{{.*}}@t27
//      CIR: cir.asm(x86_att,
// CIR-NEXT:         out = [],
// CIR-NEXT:         in = [],
// CIR-NEXT:         in_out = [],
// CIR-NEXT:         {"nop" "~{dirflag},~{fpsr},~{flags}"}) side_effects
// LLVM: define{{.*}}@t27
// LLVM: call void asm sideeffect "nop", "~{dirflag},~{fpsr},~{flags}"()
void t27(void) {
  asm volatile("nop");
}

