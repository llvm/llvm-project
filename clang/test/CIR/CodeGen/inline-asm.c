// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM,CIRLLVMONLY
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM,LLVMONLY

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

//      CIR: cir.func{{.*}}@empty3
//      CIR: %[[X:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]
//      CIR: cir.asm(x86_att, 
// CIR-NEXT:   out = [%[[X]] : !cir.ptr<!s32i> (maybe_memory)],
// CIR-NEXT:   in = [],
// CIR-NEXT:   in_out = [%[[X]] : !cir.ptr<!s32i> (maybe_memory)],
// CIR-NEXT:   {"" "=*m,*m,~{dirflag},~{fpsr},~{flags}"}) side_effects
// LLVM: define{{.*}}@empty3
// LLVM: %[[X:.*]] = alloca i32
// LLVM:   call void asm sideeffect "", "=*m,*m,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i32) %[[X]], ptr elementtype(i32) %[[X]])
void empty3(int x) {
  __asm__ volatile("" : "+m"(x));
}

//      CIR: cir.func{{.*}}@empty4
//      CIR: %[[X:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]
//      CIR: cir.asm(x86_att, 
// CIR-NEXT:   out = [],
// CIR-NEXT:   in = [%[[X]] : !cir.ptr<!s32i> (maybe_memory)],
// CIR-NEXT:   in_out = [],
// CIR-NEXT:   {"" "*m,~{dirflag},~{fpsr},~{flags}"}) side_effects
// LLVM: define{{.*}}@empty4
// LLVM: %[[X:.*]] = alloca i32
// LLVM: call void asm sideeffect "", "*m,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i32) %[[X]])
void empty4(int x) {
  __asm__ volatile("" : : "m"(x));
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

//      CIR: cir.func{{.*}}@empty6
//      CIR: %[[X:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]
//      CIR: %[[X_LOAD:.*]] = cir.load align(4) %[[X]] : !cir.ptr<!s32i>, !s32i
//      CIR: cir.asm(x86_att, 
// CIR-NEXT:   out = [],
// CIR-NEXT:   in = [],
// CIR-NEXT:   in_out = [%[[X_LOAD]] : !s32i],
// CIR-NEXT:   {"" "=&r,=&r,1,~{dirflag},~{fpsr},~{flags}"}) side_effects -> !rec_anon_struct
// LLVM: define{{.*}}@empty6
// LLVM: %[[X:.*]] = alloca i32
// LLVM: %[[X_LOAD:.*]] = load i32, ptr %[[X]]
// LLVM: call { i32, i32 } asm sideeffect "", "=&r,=&r,1,~{dirflag},~{fpsr},~{flags}"(i32 %[[X_LOAD]])
//
void empty6(int x) {
  __asm__ volatile("" : "=&r"(x), "+&r"(x));
}

//      CIR: cir.func{{.*}}@add1
//      CIR: %[[X:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["x", init]
//      CIR: %[[A:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a"]
//      CIR: %[[X_LOAD:.*]] = cir.load align(4) %[[X]] : !cir.ptr<!u32i>, !u32i
//      CIR: %[[ASM_RES:.*]] = cir.asm(x86_att, 
// CIR-NEXT:                     out = [],
// CIR-NEXT:                     in = [%[[X_LOAD]] : !u32i],
// CIR-NEXT:                     in_out = [],
// CIR-NEXT:                     {"addl $$42, $1" "=r,r,~{dirflag},~{fpsr},~{flags}"}) -> !s32i
// CIR-NEXT: cir.store{{.*}} %[[ASM_RES]], %[[A]] : !s32i, !cir.ptr<!s32i>
// LLVM: define{{.*}}@add1
// LLVM: %[[X:.*]] = alloca i32
// Classic codegen doesn't alloca for a return slot.
// CIRLLVMONLY: alloca i32
// LLVM: %[[A:.*]] = alloca i32
// LLVM: %[[X_LOAD:.*]] = load i32, ptr %[[X]]
// LLVM: %[[ASM_RES:.*]] = call i32 asm "addl $$42, $1", "=r,r,~{dirflag},~{fpsr},~{flags}"(i32 %[[X_LOAD]])
// LLVM: store i32 %[[ASM_RES]], ptr %[[A]]
unsigned add1(unsigned int x) {
  int a;
  __asm__("addl $42, %[val]"
      : "=r" (a)
      : [val] "r" (x)
      );

  return a;
}

//      CIR: cir.func{{.*}}@add2
//      CIR: %[[X:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["x", init]
//      CIR: %[[X_LOAD:.*]] = cir.load align(4) %[[X]] : !cir.ptr<!u32i>, !u32i
//      CIR: %[[ASM_RES:.*]] = cir.asm(x86_att, 
// CIR-NEXT:                     out = [],
// CIR-NEXT:                     in = [],
// CIR-NEXT:                     in_out = [%[[X_LOAD]] : !u32i],
// CIR-NEXT:                     {"addl $$42, $0" "=r,0,~{dirflag},~{fpsr},~{flags}"}) -> !u32i
// CIR-NEXT: cir.store{{.*}} %[[ASM_RES]], %[[X]] : !u32i, !cir.ptr<!u32i>
// LLVM: define{{.*}}@add2
// LLVM: %[[X:.*]] = alloca i32
// LLVM: %[[X_LOAD:.*]] = load i32, ptr %[[X]]
// LLVM: %[[ASM_RES:.*]] = call i32 asm "addl $$42, $0", "=r,0,~{dirflag},~{fpsr},~{flags}"(i32 %[[X_LOAD]])
// LLVM: store i32 %[[ASM_RES]], ptr %[[X]]
unsigned add2(unsigned int x) {
  __asm__("addl $42, %[val]"
      : [val] "+r" (x)
      );
  return x;
}

//      CIR: cir.func{{.*}}@add3
//      CIR: %[[X:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["x", init]
//      CIR: %[[X_LOAD:.*]] = cir.load{{.*}} %[[X]] : !cir.ptr<!u32i>, !u32i
//      CIR: %[[ASM_RES:.*]] = cir.asm(x86_att, 
// CIR-NEXT:                     out = [],
// CIR-NEXT:                     in = [],
// CIR-NEXT:                     in_out = [%[[X_LOAD]] : !u32i],
// CIR-NEXT:                     {"addl $$42, $0  \0A\09          subl $$1, $0    \0A\09          imul $$2, $0" "=r,0,~{dirflag},~{fpsr},~{flags}"}) -> !u32i
// CIR-NEXT: cir.store{{.*}} %[[ASM_RES]], %[[X]]  : !u32i, !cir.ptr<!u32i>
// LLVM: define{{.*}}@add3
// LLVM: %[[X:.*]] = alloca i32
// LLVM: %[[X_LOAD:.*]] = load i32, ptr %[[X]]
// LLVM: %[[ASM_RES:.*]] = call i32 asm "addl $$42, $0  \0A\09          subl $$1, $0    \0A\09          imul $$2, $0", "=r,0,~{dirflag},~{fpsr},~{flags}"(i32 %[[X_LOAD]])
// LLVM: store i32 %[[ASM_RES]], ptr %[[X]]
unsigned add3(unsigned int x) { // ((42 + x) - 1) * 2
  __asm__("addl $42, %[val]  \n\t\
          subl $1, %[val]    \n\t\
          imul $2, %[val]"
      : [val] "+r" (x)
      );
  return x;
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

//      CIR: cir.func{{.*}}@add5
//      CIR: %[[X:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["x", init]
//      CIR: %[[Y:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["y", init]
//      CIR: %[[R:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["r"]
//      CIR: %[[X_LOAD:.*]] = cir.load{{.*}} %[[X]] : !cir.ptr<!cir.float>, !cir.float
//      CIR: %[[Y_LOAD:.*]] = cir.load{{.*}} %[[Y]] : !cir.ptr<!cir.float>, !cir.float
//      CIR: %[[ASM_RES:.*]] = cir.asm(x86_att, 
// CIR-NEXT:                      out = [],
// CIR-NEXT:                      in = [%[[X_LOAD]] : !cir.float, %[[Y_LOAD]] : !cir.float],
// CIR-NEXT:                      in_out = [],
// CIR-NEXT:                      {"flds $1; flds $2; faddp" "=&{st},imr,imr,~{dirflag},~{fpsr},~{flags}"}) -> !cir.float
// CIR-NEXT: cir.store{{.*}} %[[ASM_RES]], %[[R]] : !cir.float, !cir.ptr<!cir.float>
// LLVM: define{{.*}}@add5
// LLVM: %[[X:.*]] = alloca float
// LLVM: %[[Y:.*]] = alloca float
// Classic codegen doesn't alloca for a return slot.
// CIRLLVMONLY: alloca float
// LLVM: %[[R:.*]] = alloca float
// LLVM: %[[X_LOAD:.*]] = load float, ptr %[[X]]
// LLVM: %[[Y_LOAD:.*]] = load float, ptr %[[Y]]
// LLVM: %[[ASM_RES:.*]] = call float asm "flds $1; flds $2; faddp", "=&{st},imr,imr,~{dirflag},~{fpsr},~{flags}"(float %[[X_LOAD]], float %[[Y_LOAD]])
// LLVM: store float %[[ASM_RES]], ptr %[[R]]
float add5(float x, float y) {
   float r;
  __asm__("flds %[x]; flds %[y]; faddp"
          : "=&t" (r)
          : [x] "g" (x), [y] "g" (y));
  return r;
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

//      CIR: cir.func{{.*}}@t1
//      CIR: %[[LEN:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["len", init]
//      CIR: %[[ASM_STRUCT:.*]] = cir.alloca !rec_anon_struct, !cir.ptr<!rec_anon_struct>, ["__asm_result"]
//      CIR: %[[LEN_LOAD:.*]] = cir.load align(4) %[[LEN]] : !cir.ptr<!s32i>, !s32i
//      CIR: %[[ASM_RES:.*]] = cir.asm(x86_att,
// CIR-NEXT:                         out = [],
// CIR-NEXT:                         in = [],
// CIR-NEXT:                         in_out = [%[[LEN_LOAD:.*]] : !s32i],
// CIR-NEXT:                         {"" "=&r,=&r,1,~{dirflag},~{fpsr},~{flags}"}) side_effects -> !rec_anon_struct
// CIR-NEXT: cir.store align(1) %[[ASM_RES]], %[[ASM_STRUCT]]
// CIR-NEXT: %[[GET_MEM:.*]] = cir.get_member %[[ASM_STRUCT]][0] {name = ""} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!s32i>
// CIR-NEXT: cir.load align(1) %[[GET_MEM]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[GET_MEM:.*]] = cir.get_member %[[ASM_STRUCT]][1] {name = ""} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!s32i>
// CIR-NEXT: cir.load align(1) %[[GET_MEM]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: cir.store 
// CIR-NEXT: cir.store
// CIR-NEXT: cir.return
// LLVM: define{{.*}}@t1
// LLVM: %[[LEN:.*]] = alloca i32
// LLVM: %[[LEN_LOAD:.*]] = load i32, ptr %[[LEN]]
// LLVM: call { i32, i32 } asm sideeffect "", "=&r,=&r,1,~{dirflag},~{fpsr},~{flags}"(i32 %[[LEN_LOAD]])
void t1(int len) {
  __asm__ volatile("" : "=&r"(len), "+&r"(len));
}

//      CIR: cir.func{{.*}}@t2
//      CIR: %[[T:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["t", init]
//      CIR: cir.asm(x86_att,
// CIR-NEXT:         out = [%[[T]] : !cir.ptr<!u64i> (maybe_memory)],
// CIR-NEXT:         in = [],
// CIR-NEXT:         in_out = [%[[T]] : !cir.ptr<!u64i> (maybe_memory)],
// CIR-NEXT:         {"" "=*m,*m,~{dirflag},~{fpsr},~{flags}"}) side_effects
// LLVM: define{{.*}}@t2
// LLVM: %[[T:.*]] = alloca i64
// LLVM: call void asm sideeffect "", "=*m,*m,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i64) %[[T]], ptr elementtype(i64) %[[T]])
void t2(unsigned long long t)  {
  __asm__ volatile("" : "+m"(t));
}

//      CIR: cir.func{{.*}}@t3
//      CIR: %[[SRC:.*]] = cir.alloca !cir.ptr<!u8i>, !cir.ptr<!cir.ptr<!u8i>>, ["src", init]
// CIR-NEXT: %[[TMP:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["temp", init]
// CIR-NEXT: cir.store
// CIR-NEXT: cir.store
// CIR-NEXT: %[[SRC_LOAD:.*]] = cir.load align(8) %[[SRC]] : !cir.ptr<!cir.ptr<!u8i>>, !cir.ptr<!u8i>
// CIR-NEXT: cir.asm(x86_att,
// CIR-NEXT:         out = [%[[TMP]] : !cir.ptr<!u64i> (maybe_memory)],
// CIR-NEXT:         in = [],
// CIR-NEXT:         in_out = [%[[TMP]] : !cir.ptr<!u64i> (maybe_memory), %[[SRC_LOAD]] : !cir.ptr<!u8i>],
// CIR-NEXT:         {"" "=*m,=r,*m,1,~{dirflag},~{fpsr},~{flags}"}) side_effects -> !cir.ptr<!u8i>
// LLVM: define{{.*}}@t3
// LLVM: %[[SRC_ADDR:.*]] = alloca ptr
// LLVM: %[[TMP:.*]] = alloca i64
// LLVM: %[[SRC:.*]] = load ptr, ptr %[[SRC_ADDR]]
// LLVM: call ptr asm sideeffect "", "=*m,=r,*m,1,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i64) %[[TMP]], ptr elementtype(i64) %[[TMP]], ptr %[[SRC]])
void t3(unsigned char *src, unsigned long long temp) {
  __asm__ volatile("" : "+m"(temp), "+r"(src));
}

//      CIR: cir.func{{.*}}@t4
//      CIR: %[[A:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["a"] {alignment = 8 : i64}
//      CIR: %[[B:.*]] = cir.alloca !rec_reg, !cir.ptr<!rec_reg>, ["b"] {alignment = 8 : i64}
//      CIR: cir.asm(x86_att, 
// CIR-NEXT:   out = [],
// CIR-NEXT:   in = [%[[A]] : !cir.ptr<!u64i> (maybe_memory), %[[B]] : !cir.ptr<!rec_reg> (maybe_memory)],
// CIR-NEXT:   in_out = [],
// CIR-NEXT:   {"" "*m,*m,~{dirflag},~{fpsr},~{flags}"}) side_effects
// LLVM: define{{.*}}@t4
// LLVM: %[[A:.*]] = alloca i64
// LLVM: %[[B:.*]] = alloca %struct.reg
// LLVM: call void asm sideeffect "", "*m,*m,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i64) %[[A]], ptr elementtype(%struct.reg) %[[B]])
void t4(void) {
  unsigned long long a;
  struct reg { unsigned long long a, b; } b;

  __asm__ volatile ("":: "m"(a), "m"(b));
}

//      CIR: cir.func{{.*}}@t5
//      CIR: %[[I:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init]
//      CIR: %[[FUNC:.*]] = cir.get_global @t5 : !cir.ptr<!cir.func<(!s32i)>>
//      CIR: %[[ASM_RES:.*]] = cir.asm(x86_att, 
// CIR-NEXT:                     out = [],
// CIR-NEXT:                     in = [%[[FUNC]] : !cir.ptr<!cir.func<(!s32i)>>],
// CIR-NEXT:                     in_out = [],
// CIR-NEXT:                     {"nop" "=r,0,~{dirflag},~{fpsr},~{flags}"}) -> !cir.ptr<!cir.func<(!s32i)>>
//      CIR: %[[P_TO_I:.*]] = cir.cast ptr_to_int %[[ASM_RES]] : !cir.ptr<!cir.func<(!s32i)>> -> !u64i
//      CIR: %[[INT_CAST:.*]] = cir.cast integral %[[P_TO_I]] : !u64i -> !s32i
//      CIR: cir.store align(4) %[[INT_CAST]], %[[I]] : !s32i, !cir.ptr<!s32i>
// LLVM: define{{.*}}@t5
// LLVM: %[[I:.*]] = alloca i32
// LLVM: %[[ASM_RES:.*]] = call ptr asm "nop", "=r,0,~{dirflag},~{fpsr},~{flags}"(ptr @t5)
// LLVM: %[[ASM_RES_CAST:.*]] = ptrtoint ptr %[[ASM_RES]] to i64
// LLVM: %[[ASM_RES_TRUNC:.*]] = trunc i64 %[[ASM_RES_CAST]] to i32
// LLVM: store i32 %[[ASM_RES_TRUNC]], ptr %[[I]]
void t5(int i) {
  asm("nop" : "=r"(i) : "0"(t5));
}

//      CIR: cir.func{{.*}}@t6
//      CIR: %[[FUNC:.*]] = cir.get_global @t6 : !cir.ptr<!cir.func<()>>
//      CIR: cir.asm(x86_att, 
// LLVM: define{{.*}}@t6
// CIR-NEXT:   out = [],
// CIR-NEXT:   in = [%[[FUNC]] : !cir.ptr<!cir.func<()>>],
// CIR-NEXT:   in_out = [],
// CIR-NEXT:   {"" "i,~{dirflag},~{fpsr},~{flags}"}) side_effects
// LLVM: call void asm sideeffect "", "i,~{dirflag},~{fpsr},~{flags}"(ptr @t6)
void t6(void) {
  __asm__ volatile("" : : "i" (t6));
}

//      CIR: cir.func{{.*}}@t7
//      CIR: %[[A:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
//      CIR: %[[A_LOAD:.*]] = cir.load align(4) %0 : !cir.ptr<!s32i>, !s32i
//      CIR: %[[FOUR:.*]] = cir.const #cir.int<4> : !s32i
// CIR-NEXT: %[[ASM_RES:.*]] = cir.asm(x86_att, 
// CIR-NEXT:                     out = [],
// CIR-NEXT:                     in = [%[[FOUR]] : !s32i],
// CIR-NEXT:                     in_out = [%[[A_LOAD]] : !s32i],
// CIR-NEXT:                     {"T7 NAMED: $1" "=r,i,0,~{dirflag},~{fpsr},~{flags}"}) side_effects -> !s32i
//      CIR: cir.store align(4) %[[ASM_RES]], %[[A]] : !s32i, !cir.ptr<!s32i>
// LLVM: define{{.*}}@t7
// LLVM: %[[A:.*]] = alloca i32
// LLVM: %[[A_LOAD:.*]] = load i32, ptr %[[A]]
// LLVM: %[[ASM_RES:.*]] = call i32 asm sideeffect "T7 NAMED: $1", "=r,i,0,~{dirflag},~{fpsr},~{flags}"(i32 4, i32 %[[A_LOAD]])
// LLVM: store i32 %[[ASM_RES]], ptr %[[A]]
void t7(int a) {
  __asm__ volatile("T7 NAMED: %[input]" : "+r"(a): [input] "i" (4));  
}

//      CIR: cir.func{{.*}}@t8
//      CIR: %[[FOUR:.*]] = cir.const #cir.int<4> : !s32i 
// CIR-NEXT: cir.asm(x86_att, 
// CIR-NEXT:   out = [],
// CIR-NEXT:   in = [%[[FOUR]] : !s32i],
// CIR-NEXT:   in_out = [],
// CIR-NEXT:   {"T8 NAMED MODIFIER: ${0:c}" "i,~{dirflag},~{fpsr},~{flags}"}) side_effects
// LLVM: define{{.*}}@t8
// LLVM: call void asm sideeffect "T8 NAMED MODIFIER: ${0:c}", "i,~{dirflag},~{fpsr},~{flags}"(i32 4)
void t8(void) {
  __asm__ volatile("T8 NAMED MODIFIER: %c[input]" :: [input] "i" (4));
}

//      CIR: cir.func{{.*}}@t9
//      CIR: %[[A:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["a", init]
//      CIR: %[[A_LOAD:.*]] = cir.load align(4) %[[A]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: cir.asm(x86_att,
// CIR-NEXT:         out = [],
// CIR-NEXT:         in = [],
// CIR-NEXT:         in_out = [%[[A_LOAD]] : !u32i],
// CIR-NEXT:         {"bswap $0 $1" "=r,0,~{dirflag},~{fpsr},~{flags}"}) -> !u32i
// LLVM: define{{.*}}@t9
// LLVM: %[[A_ADDR:.*]] = alloca i32
// LLVM: %[[A:.*]] = load i32, ptr %[[A_ADDR]]
// LLVM: call i32 asm "bswap $0 $1", "=r,0,~{dirflag},~{fpsr},~{flags}"(i32 %[[A]])
unsigned t9(unsigned int a) {
  asm("bswap %0 %1" : "+r" (a));
  return a;
}

//      CIR: cir.func{{.*}}@t10
//      CIR: %[[R:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["r", init]
//      CIR: %[[R_LOAD:.*]] = cir.load align(4) %[[R]] : !cir.ptr<!s32i>, !s32i
//      CIR: %[[ZERO1:.*]] = cir.const #cir.int<0> : !s32i
//      CIR: %[[ZERO2:.*]] = cir.const #cir.int<0> : !s32i
//      CIR: %[[ZERO3:.*]] = cir.const #cir.int<0> : !s32i
//      CIR: %[[ZERO_3_CAST:.*]] = cir.cast int_to_float %[[ZERO3]] : !s32i -> !cir.double
// CIR-NEXT: %[[ASM_RES:.*]] = cir.asm(x86_att, 
// CIR-NEXT:                     out = [],
// CIR-NEXT:                     in = [%[[ZERO1]] : !s32i, %[[ZERO2]] : !s32i, %[[ZERO_3_CAST]] : !cir.double],
// CIR-NEXT:                     in_out = [%[[R_LOAD]] : !s32i],
// CIR-NEXT:                     {"PR3908 $1 $3 $2 $0" "=r,mx,mr,x,0,~{dirflag},~{fpsr},~{flags}"}) -> !s32i
// CIR-NEXT: cir.store align(4) %[[ASM_RES]], %[[R]] : !s32i, !cir.ptr<!s32i>
// LLVM: define{{.*}}@t10
// LLVM: %[[R:.*]] = alloca i32
// LLVM: %[[R_LOAD:.*]] = load i32, ptr %[[R]]
// LLVM: %[[ASM_RES:.*]] = call i32 asm "PR3908 $1 $3 $2 $0", "=r,mx,mr,x,0,~{dirflag},~{fpsr},~{flags}"(i32 0, i32 0, double 0{{.*}}, i32 %[[R_LOAD]])
// LLVM: store i32 %[[ASM_RES]], ptr %[[R]]
void t10(int r) {
  __asm__("PR3908 %[lf] %[xx] %[li] %[r]" : [r] "+r" (r) : [lf] "mx" (0), [li] "mr" (0), [xx] "x" ((double)(0)));
}

//      CIR: cir.func{{.*}}@t11
//      CIR: %[[INPUT:.*]] = cir.alloca !s8i, !cir.ptr<!s8i>, ["input", init]
//      CIR: %[[OUTPUT:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["output"]
//      CIR: %[[INPUT_LOAD:.*]] = cir.load align(1) %[[INPUT]] : !cir.ptr<!s8i>, !s8i
//      CIR: %[[INPUT_CAST:.*]] = cir.cast integral %[[INPUT_LOAD]] : !s8i -> !u32i
// CIR-NEXT: %[[ASM_RES:.*]] = cir.asm(x86_att, 
// CIR-NEXT:                     out = [],
// CIR-NEXT:                     in = [%[[INPUT_CAST]] : !u32i],
// CIR-NEXT:                     in_out = [],
// CIR-NEXT:                     {"xyz" "={ax},0,~{dirflag},~{fpsr},~{flags}"}) -> !u32i
// CIR-NEXT: cir.store align(4) %[[ASM_RES]], %[[OUTPUT]] : !u32i, !cir.ptr<!u32i>
// LLVM: define{{.*}}@t11
// LLVM: %[[INPUT:.*]] = alloca i8
// Classic codegen doesn't alloca for a return slot.
// CIRLLVMONLY: alloca i32
// LLVM: %[[OUTPUT:.*]] = alloca i32
// LLVM: %[[INPUT_LOAD:.*]] = load i8, ptr %[[INPUT]]
// The two IR forms disagree as to whether this should be a zero or sign extend.
// It looks like this should be a 'sign' extend (sign->unsigned cast typically
// results in a sext), so this might be a bug in classic codegen.
// CIRLLVMONLY: %[[INPUT_EXT:.*]] = sext i8 %[[INPUT_LOAD]] to i32
// LLVMONLY: %[[INPUT_EXT:.*]] = zext i8 %[[INPUT_LOAD]] to i32
// LLVM: %[[ASM_RES:.*]] = call i32 asm "xyz", "={ax},0,~{dirflag},~{fpsr},~{flags}"(i32 %[[INPUT_EXT]])
// LLVM: store i32 %[[ASM_RES]], ptr %[[OUTPUT]]
unsigned t11(signed char input) {
  unsigned  output;
  __asm__("xyz"
          : "=a" (output)
          : "0" (input));
  return output;
}

//      CIR: cir.func{{.*}}@t12
//      CIR: %[[INPUT:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["input", init]
//      CIR: %[[OUTPUT:.*]] = cir.alloca !u8i, !cir.ptr<!u8i>, ["output"]
//      CIR: %[[INPUT_LOAD:.*]] = cir.load align(4) %[[INPUT]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: %[[ASM_RES:.*]] = cir.asm(x86_att, 
// CIR-NEXT:                     out = [],
// CIR-NEXT:                     in = [%[[INPUT_LOAD]] : !u32i],
// CIR-NEXT:                     in_out = [],
// CIR-NEXT:                     {"xyz" "={ax},0,~{dirflag},~{fpsr},~{flags}"}) -> !u32i
// CIR-NEXT: %[[ASM_RES_INT:.*]] = cir.cast integral %[[ASM_RES]] : !u32i -> !u8i
// CIR-NEXT: cir.store align(1) %[[ASM_RES_INT]], %[[OUTPUT]] : !u8i, !cir.ptr<!u8i>
// LLVM: define{{.*}}@t12
// LLVM: %[[INPUT:.*]] = alloca i32
// Classic codegen doesn't alloca for a return slot.
// CIRLLVMONLY: alloca i8
// LLVM: %[[OUTPUT:.*]] = alloca i8
// LLVM: %[[INPUT_LOAD:.*]] = load i32, ptr %[[INPUT]]
// LLVM: %[[ASM_RES:.*]] = call i32 asm "xyz", "={ax},0,~{dirflag},~{fpsr},~{flags}"(i32 %[[INPUT_LOAD]])
// LLVM: %[[ASM_RES_TRUNC:.*]] = trunc i32 %[[ASM_RES]] to i8
// LLVM: store i8 %[[ASM_RES_TRUNC]], ptr %[[OUTPUT]]
unsigned char t12(unsigned input) {
  unsigned char output;
  __asm__("xyz"
          : "=a" (output)
          : "0" (input));
  return output;
}

//      CIR: cir.func{{.*}}@t13
//      CIR: %[[INPUT:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["input", init]
//      CIR: %[[OUTPUT:.*]] = cir.alloca !u8i, !cir.ptr<!u8i>, ["output"]
//      CIR: %[[INPUT_LOAD:.*]] = cir.load align(4) %[[INPUT]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: %[[ASM_RES:.*]] = cir.asm(x86_att, 
// CIR-NEXT:                     out = [],
// CIR-NEXT:                     in = [%[[INPUT_LOAD]] : !u32i],
// CIR-NEXT:                     in_out = [],
// CIR-NEXT:                     {"xyz $1" "={ax},0,~{dirflag},~{fpsr},~{flags}"}) -> !u32i
// CIR-NEXT: %[[ASM_RES_INT:.*]] = cir.cast integral %[[ASM_RES]] : !u32i -> !u8i
// CIR-NEXT: cir.store align(1) %[[ASM_RES_INT]], %[[OUTPUT]] : !u8i, !cir.ptr<!u8i>
// LLVM: define{{.*}}@t13
// LLVM: %[[INPUT:.*]] = alloca i32
// Classic codegen doesn't alloca for a return slot.
// CIRLLVMONLY: alloca i8
// LLVM: %[[OUTPUT:.*]] = alloca i8
// LLVM: %[[INPUT_LOAD:.*]] = load i32, ptr %[[INPUT]]
// LLVM: %[[ASM_RES:.*]] = call i32 asm "xyz $1", "={ax},0,~{dirflag},~{fpsr},~{flags}"(i32 %[[INPUT_LOAD]])
// LLVM: %[[ASM_RES_TRUNC:.*]] = trunc i32 %[[ASM_RES]] to i8
// LLVM: store i8 %[[ASM_RES_TRUNC]], ptr %[[OUTPUT]]
unsigned char t13(unsigned input) {
  unsigned char output;
  __asm__("xyz %1"
          : "=a" (output)
          : "0" (input));
  return output;
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

//      CIR: cir.func{{.*}}@t15
//      CIR: %[[X:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]
//      CIR: %[[LARGE:.*]] = cir.alloca !cir.ptr<!rec_large>, !cir.ptr<!cir.ptr<!rec_large>>, ["P", init]
//      CIR: %[[LARGE_LOAD:.*]] = cir.load deref align(8) %1 : !cir.ptr<!cir.ptr<!rec_large>>, !cir.ptr<!rec_large>
//      CIR: %[[X_LOAD:.*]] = cir.load align(4) %0 : !cir.ptr<!s32i>, !s32i
//      CIR: %[[ASM_RES:.*]] = cir.asm(x86_att, 
// CIR-NEXT:                     out = [],
// CIR-NEXT:                     in = [%[[LARGE_LOAD]] : !cir.ptr<!rec_large> (maybe_memory), %[[X_LOAD]] : !s32i],
// CIR-NEXT:                     in_out = [],
// CIR-NEXT:                     {"xyz " "=r,*m,0,~{dirflag},~{fpsr},~{flags}"}) -> !s32i
// CIR-NEXT: cir.store align(4) %[[ASM_RES]], %[[X]] : !s32i, !cir.ptr<!s32i>
// LLVM: define{{.*}}@t15
// LLVM: %[[X:.*]] = alloca i32
// LLVM: %[[LARGE:.*]] = alloca ptr
// LLVM: %[[LARGE_LOAD:.*]] = load ptr, ptr %[[LARGE]]
// LLVM: %[[X_LOAD:.*]] = load i32, ptr %[[X]]
// LLVM: %[[ASM_RES:.*]] = call i32 asm "xyz ", "=r,*m,0,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(%struct.large) %[[LARGE_LOAD]], i32 %[[X_LOAD]])
// LLVM: store i32 %[[ASM_RES]], ptr %[[X]]
unsigned long t15(int x, struct large *P) {
  __asm__("xyz "
          : "=r" (x)
          : "m" (*P), "0" (x));
  return x;
}

//      CIR: cir.func{{.*}}@t16
//      CIR: %[[A:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a"]
//      CIR: %[[B:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b"]
//      CIR: %[[B_LOAD:.*]] = cir.load align(4) %[[B]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[ASM_RES:.*]] = cir.asm(x86_att, 
// CIR-NEXT:                     out = [],
// CIR-NEXT:                     in = [%[[B_LOAD]] : !s32i],
// CIR-NEXT:                     in_out = [],
// CIR-NEXT:                     {"nop;" "=%{cx},r,~{dirflag},~{fpsr},~{flags}"}) -> !s32i
// CIR-NEXT: cir.store align(4) %[[ASM_RES]], %[[A]] : !s32i, !cir.ptr<!s32i>
// LLVM: define{{.*}}@t16
// Classic codegen doesn't alloca for a return slot.
// CIRLLVMONLY: alloca i32
// LLVM: %[[A:.*]] = alloca i32
// LLVM: %[[B:.*]] = alloca i32
// LLVM: %[[B_LOAD:.*]] = load i32, ptr %[[B]]
// LLVM: %[[ASM_RES:.*]] = call i32 asm "nop;", "=%{cx},r,~{dirflag},~{fpsr},~{flags}"(i32 %[[B_LOAD]])
// LLVM: store i32 %[[ASM_RES]], ptr %[[A]]
int t16(void) {
  int a,b;
  asm ( "nop;"
       :"=%c" (a)
       : "r" (b)
       );
  return 0;
}

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

//      CIR: cir.func{{.*}}@t18
//      CIR: %[[DATA:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["data", init]
//      CIR: %[[A:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a"]
//      CIR: %[[B:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b"]
//      CIR: %[[ASM_RES_VAR:.*]] = cir.alloca !rec_anon_struct, !cir.ptr<!rec_anon_struct>, ["__asm_result"]
//      CIR: %[[DATA_LOAD:.*]] = cir.load align(4) %[[DATA]] : !cir.ptr<!u32i>, !u32i
//      CIR: %[[ASM_RES:.*]] = cir.asm(x86_att, 
// CIR-NEXT:                     out = [],
// CIR-NEXT:                     in = [%[[DATA_LOAD]] : !u32i],
// CIR-NEXT:                     in_out = [],
// CIR-NEXT:                     {"xyz" "={ax},={dx},{ax},~{dirflag},~{fpsr},~{flags}"}) -> !rec_anon_struct
// CIR-NEXT: cir.store align(1) %[[ASM_RES]], %[[ASM_RES_VAR]] : !rec_anon_struct, !cir.ptr<!rec_anon_struct>
// CIR-NEXT: %[[GM_FIRST:.*]] = cir.get_member %[[ASM_RES_VAR]][0] {name = ""} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!s32i>
// CIR-NEXT: %[[GM_FIRST_LOAD:.*]] = cir.load align(1) %[[GM_FIRST]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[GM_SEC:.*]] = cir.get_member %[[ASM_RES_VAR]][1] {name = ""} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!s32i>
// CIR-NEXT: %[[GM_SEC_LOAD:.*]] = cir.load align(1) %[[GM_SEC]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: cir.store align(4) %[[GM_FIRST_LOAD]], %[[A]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: cir.store align(4) %[[GM_SEC_LOAD]], %[[B]] : !s32i, !cir.ptr<!s32i>
// LLVM: define{{.*}}@t18
// LLVM: %[[DATA:.*]] = alloca i32
// Classic codegen doesn't alloca for a return slot.
// CIRLLVMONLY: alloca i32
// LLVM: %[[A:.*]] = alloca i32
// LLVM: %[[B:.*]] = alloca i32
//
// Classic codegen doesn't alloca a slot for this, and uses extractvalue instead.
// CIRLLVMONLY: %[[ASM_RES_VAR:.*]] = alloca { i32, i32 }

// LLVM: %[[DATA_LOAD:.*]] = load i32, ptr %[[DATA]]
// LLVM: %[[ASM_RES:.*]] = call { i32, i32 } asm "xyz", "={ax},={dx},{ax},~{dirflag},~{fpsr},~{flags}"(i32 %[[DATA_LOAD]])
//
// Because we go through get_member, the codegen here is slightly different.
// CIRLLVMONLY: store { i32, i32 } %[[ASM_RES]], ptr %[[ASM_RES_VAR]]
// CIRLLVMONLY: %[[GEP_FIRST:.*]] = getelementptr { i32, i32 }, ptr %[[ASM_RES_VAR]], i32 0, i32 0
// CIRLLVMONLY: %[[GEP_FIRST_LOAD:.*]] = load i32, ptr %[[GEP_FIRST]]
// CIRLLVMONLY: %[[GEP_SECOND:.*]] = getelementptr { i32, i32 }, ptr %[[ASM_RES_VAR]], i32 0, i32 1
// CIRLLVMONLY: %[[GEP_SECOND_LOAD:.*]] = load i32, ptr %[[GEP_SECOND]]
//
// LLVMONLY: %[[GEP_FIRST_LOAD:.*]] = extractvalue { i32, i32 } %[[ASM_RES]], 0
// LLVMONLY: %[[GEP_SECOND_LOAD:.*]] = extractvalue { i32, i32 } %[[ASM_RES]], 1
// 
// LLVM: store i32 %[[GEP_FIRST_LOAD]], ptr %[[A]]
// LLVM: store i32 %[[GEP_SECOND_LOAD]], ptr %[[B]]
int t18(unsigned data) {
  int a, b;

  asm("xyz" :"=a"(a), "=d"(b) : "a"(data));
  return a + b;
}

//      CIR: cir.func{{.*}}@t19
//      CIR: %[[DATA:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["data", init]
//      CIR: %[[A:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a"]
//      CIR: %[[B:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b"]
//      CIR: %[[DATA_LOAD:.*]] = cir.load align(4) %[[DATA]] : !cir.ptr<!u32i>, !u32i
//      CIR: %[[ASM_RES:.*]] = cir.asm(x86_att, 
// CIR-NEXT:                     out = [],
// CIR-NEXT:                     in = [%[[DATA_LOAD]] : !u32i],
// CIR-NEXT:                     in_out = [],
// CIR-NEXT:                     {"x$(abc$|def$|ghi$)z" "=r,r,~{dirflag},~{fpsr},~{flags}"}) -> !s32i
// CIR-NEXT: cir.store align(4) %[[ASM_RES]], %[[A]] : !s32i, !cir.ptr<!s32i>
// LLVM: define{{.*}}@t19
// LLVM: %[[DATA:.*]] = alloca i32
// Classic codegen doesn't alloca for a return slot.
// CIRLLVMONLY: alloca i32
// LLVM: %[[A:.*]] = alloca i32
// LLVM: %[[B:.*]] = alloca i32
// LLVM: %[[DATA_LOAD:.*]] = load i32, ptr %[[DATA]]
// LLVM: %[[ASM_RES:.*]] = call i32 asm "x$(abc$|def$|ghi$)z", "=r,r,~{dirflag},~{fpsr},~{flags}"(i32 %[[DATA_LOAD]])
// LLVM: store i32 %[[ASM_RES]], ptr %[[A]]
int t19(unsigned data) {
  int a, b;

  asm("x{abc|def|ghi}z" :"=r"(a): "r"(data));
  return a + b;
}

// skip t20 and t21: long double is not supported
//      CIR: cir.func{{.*}}@t22
//      CIR: %[[LA:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["la", init]
//      CIR: %[[LB:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["lb", init]
//      CIR: %[[BIGRES:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["bigres"]
//      CIR: %[[LA_LOAD:.*]] = cir.load align(4) %[[LA]] : !cir.ptr<!u32i>, !u32i
//      CIR: %[[LB_LOAD:.*]] = cir.load align(4) %[[LB]] : !cir.ptr<!u32i>, !u32i
//      CIR: %[[ASM_RES:.*]] = cir.asm(x86_att, 
// CIR-NEXT:                     out = [],
// CIR-NEXT:                     in = [%[[LA_LOAD]] : !u32i, %[[LB_LOAD]] : !u32i],
// CIR-NEXT:                     in_out = [],
// CIR-NEXT:                     {"0:\0A1:\0A" "=l{ax},0,{cx},~{edx},~{cc},~{dirflag},~{fpsr},~{flags}"}) -> !u32i
// CIR-NEXT: cir.store align(4) %[[ASM_RES]], %[[BIGRES]] : !u32i, !cir.ptr<!u32i>
// LLVM: define{{.*}}@t22
// LLVM: %[[LA:.*]] = alloca i32
// LLVM: %[[LB:.*]] = alloca i32
// LLVM: %[[BIGRES:.*]] = alloca i32
// LLVM: %[[LA_LOAD:.*]] = load i32, ptr %[[LA]]
// LLVM: %[[LB_LOAD:.*]] = load i32, ptr %[[LB]]
// LLVM: %[[ASM_RES:.*]] = call i32 asm "0:\0A1:\0A", "=l{ax},0,{cx},~{edx},~{cc},~{dirflag},~{fpsr},~{flags}"(i32 %[[LA_LOAD]], i32 %[[LB_LOAD]])
// LLVM: store i32 %[[ASM_RES]], ptr %[[BIGRES]]
// accept 'l' constraint
unsigned char t22(unsigned char a, unsigned char b) {
  unsigned int la = a;
  unsigned int lb = b;
  unsigned int bigres;
  unsigned char res;
  __asm__ ("0:\n1:\n" : [bigres] "=la"(bigres) : [la] "0"(la), [lb] "c"(lb) :
                        "edx", "cc");
  res = bigres;
  return res;
}

//      CIR: cir.func{{.*}}@t23
//      CIR: %[[LA:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["la", init]
//      CIR: %[[LB:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["lb", init]
//      CIR: %[[RES:.*]] = cir.alloca !u8i, !cir.ptr<!u8i>, ["res"]
//      CIR: %[[LA_LOAD:.*]] = cir.load align(4) %[[LA]] : !cir.ptr<!u32i>, !u32i
//      CIR: %[[LB_LOAD:.*]] = cir.load align(4) %[[LB]] : !cir.ptr<!u32i>, !u32i
//      CIR: %[[ASM_RES:.*]] = cir.asm(x86_att, 
// CIR-NEXT:                     out = [],
// CIR-NEXT:                     in = [%[[LA_LOAD]] : !u32i, %[[LB_LOAD]] : !u32i],
// CIR-NEXT:                     in_out = [],
// CIR-NEXT:                     {"0:\0A1:\0A" "=l{ax},0,{cx},~{edx},~{cc},~{dirflag},~{fpsr},~{flags}"}) -> !u32i
// CIR-NEXT: %[[ASM_RES_CAST:.*]] = cir.cast integral %[[ASM_RES]] : !u32i -> !u8i
// CIR-NEXT: cir.store align(1) %[[ASM_RES_CAST]], %[[RES]] : !u8i, !cir.ptr<!u8i>
// LLVM: define{{.*}}@t23
// LLVM: %[[LA:.*]] = alloca i32
// LLVM: %[[LB:.*]] = alloca i32
// LLVM: %[[RES:.*]] = alloca i8
// LLVM: %[[LA_LOAD:.*]] = load i32, ptr %[[LA]]
// LLVM: %[[LB_LOAD:.*]] = load i32, ptr %[[LB]]
// LLVM: %[[ASM_RES:.*]] = call i32 asm "0:\0A1:\0A", "=l{ax},0,{cx},~{edx},~{cc},~{dirflag},~{fpsr},~{flags}"(i32 %[[LA_LOAD]], i32 %[[LB_LOAD]])
// LLVM: %[[ASM_RES_TRUNC:.*]] = trunc i32 %[[ASM_RES]] to i8
// LLVM: store i8 %[[ASM_RES_TRUNC]], ptr %[[RES]]
// accept 'l' constraint
unsigned char t23(unsigned char a, unsigned char b) {
  unsigned int la = a;
  unsigned int lb = b;
  unsigned char res;
  __asm__ ("0:\n1:\n" : [res] "=la"(res) : [la] "0"(la), [lb] "c"(lb) :
                        "edx", "cc");
  return res;
}

//      CIR: cir.func{{.*}}@t24
//      CIR: %[[C:.*]] = cir.alloca !s8i, !cir.ptr<!s8i>, ["c", init]
//      CIR: %[[ADDR:.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["addr"]
//      CIR: %[[C_LOAD:.*]] = cir.load align(1) %[[C]] : !cir.ptr<!s8i>, !s8i
//      CIR: %[[C_LOAD_CAST:.*]] = cir.cast integral %[[C_LOAD]] : !s8i -> !u64i
//      CIR: %[[ASM_RES:.*]] = cir.asm(x86_att, 
// CIR-NEXT:                     out = [],
// CIR-NEXT:                     in = [%[[C_LOAD_CAST]] : !u64i]
// CIR-NEXT:                     in_out = [],
// CIR-NEXT:                     {"foobar" "={ax},0,~{dirflag},~{fpsr},~{flags}"}) -> !cir.ptr<!void>
// CIR-NEXT: cir.store align(8) %[[ASM_RES]], %[[ADDR]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// LLVM: define{{.*}}@t24
// LLVM: %[[C:.*]] = alloca i8
// Classic codegen doesn't alloca for a return slot.
// CIRLLVMONLY: alloca ptr
// LLVM: %[[ADDR:.*]] = alloca ptr
// LLVM: %[[C_LOAD:.*]] = load i8, ptr %[[C]]
// The two IR forms disagree as to whether this should be a zero or sign extend.
// It looks like this should be a 'sign' extend (sign->unsigned cast typically
// results in a sext), so this might be a bug in classic codegen.
// CIRLLVMONLY: %[[C_LOAD_EXT:.*]] = sext i8 %[[C_LOAD]] to i64
// LLVMONLY: %[[C_LOAD_EXT:.*]] = zext i8 %[[C_LOAD]] to i64
// LLVM: %[[ASM_RES:.*]] = call ptr asm "foobar", "={ax},0,~{dirflag},~{fpsr},~{flags}"(i64 %[[C_LOAD_EXT]])
// LLVM: store ptr %[[ASM_RES]], ptr %[[ADDR]]
void *t24(char c) {
  void *addr;
  __asm__ ("foobar" : "=a" (addr) : "0" (c));
  return addr;
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

//      CIR: cir.func{{.*}}@t28
//      CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
//      CIR: cir.asm(x86_att, 
// CIR-NEXT:   out = [],
// CIR-NEXT:   in = [%[[ONE]] : !s32i],
// CIR-NEXT:   in_out = [],
// CIR-NEXT:   {"/* $0 */" "i|r,~{dirflag},~{fpsr},~{flags}"}) side_effects
// LLVM: define{{.*}}@t28
// LLVM: call void asm sideeffect "/* $0 */", "i|r,~{dirflag},~{fpsr},~{flags}"(i32 1)
// Check handling of '*' and '#' constraint modifiers.
void t28(void)
{
  asm volatile ("/* %0 */" : : "i#*X,*r" (1));
}


static unsigned t29_var[1];

//      CIR: cir.func{{.*}}@t29
//      CIR: %[[ARR:.*]] = cir.get_global @t29_var : !cir.ptr<!cir.array<!u32i x 1>>
//      CIR: cir.asm(x86_att, 
// CIR-NEXT:   out = [],
// CIR-NEXT:   in = [%[[ARR]] : !cir.ptr<!cir.array<!u32i x 1>> (maybe_memory)],
// CIR-NEXT:   in_out = [],
// CIR-NEXT:   {"movl %eax, $0" "*m,~{dirflag},~{fpsr},~{flags}"}) side_effects
// LLVM: define{{.*}}@t29
// LLVM: call void asm sideeffect "movl %eax, $0", "*m,~{dirflag},~{fpsr},~{flags}"(ptr elementtype([1 x i32]) @t29_var)
void t29(void) {
  asm volatile("movl %%eax, %0"
               :
               : "m"(t29_var));
}

//      CIR: cir.func{{.*}}@t30
//      CIR: %[[LEN:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["len", init]
//      CIR: %[[LEN_LOAD:.*]] = cir.load align(4) %[[LEN]] : !cir.ptr<!s32i>, !s32i
//      CIR: cir.asm(x86_att,
// CIR-NEXT:         out = [%[[LEN]] : !cir.ptr<!s32i> (maybe_memory)],
// CIR-NEXT:         in = [],
// CIR-NEXT:         in_out = [%[[LEN_LOAD]] : !s32i],
// CIR-NEXT:         {"" "=*&rm,0,~{dirflag},~{fpsr},~{flags}"}) side_effects
// LLVM: define{{.*}}@t30
// LLVM: %[[LEN_ADDR:.*]] = alloca i32
// LLVM: %[[LEN:.*]] = load i32, ptr %[[LEN_ADDR]]
// LLVM: call void asm sideeffect "", "=*&rm,0,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i32) %[[LEN_ADDR]], i32 %[[LEN]])
void t30(int len) {
  __asm__ volatile(""
                   : "+&&rm"(len));
}

//      CIR: cir.func{{.*}}@t31
//      CIR: %[[LEN:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["len", init]
//      CIR: %[[LEN_LOAD:.*]] = cir.load align(4) %[[LEN]] : !cir.ptr<!s32i>, !s32i
//      CIR: %[[LEN_LOAD2:.*]] = cir.load align(4) %[[LEN]] : !cir.ptr<!s32i>, !s32i
//      CIR: cir.asm(x86_att,
// CIR-NEXT:         out = [%[[LEN]] : !cir.ptr<!s32i> (maybe_memory), %[[LEN]] : !cir.ptr<!s32i> (maybe_memory)],
// CIR-NEXT:         in = [],
// CIR-NEXT:         in_out = [%[[LEN_LOAD]] : !s32i, %[[LEN_LOAD2]] : !s32i],
// CIR-NEXT:         {"" "=*%rm,=*rm,0,1,~{dirflag},~{fpsr},~{flags}"}) side_effects
// LLVM: define{{.*}}@t31
// LLVM: %[[LEN_ADDR:.*]] = alloca i32
// LLVM: %[[LEN1:.*]] = load i32, ptr %[[LEN_ADDR]]
// LLVM: %[[LEN2:.*]] = load i32, ptr %[[LEN_ADDR]]
// LLVM: call void asm sideeffect "", "=*%rm,=*rm,0,1,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i32) %[[LEN_ADDR]], ptr elementtype(i32) %[[LEN_ADDR]], i32 %[[LEN1]], i32 %[[LEN2]])
void t31(int len) {
  __asm__ volatile(""
                   : "+%%rm"(len), "+rm"(len));
}

//t32 skipped: no goto

//      CIR: cir.func{{.*}}@t33
//      CIR: %[[PTR:.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["ptr", init]
//      CIR: %[[RET:.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["ret"]
//      CIR: %[[PTR_LOAD:.*]] = cir.load align(8) %0 : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
//      CIR: %[[ASM_RES:.*]] = cir.asm(x86_att, 
// CIR-NEXT:                     out = [],
// CIR-NEXT:                     in = [%[[PTR_LOAD]] : !cir.ptr<!void>],
// CIR-NEXT:                     in_out = [],
// CIR-NEXT:                     {"lea $1, $0" "=r,p,~{dirflag},~{fpsr},~{flags}"}) -> !cir.ptr<!void>
// CIR-NEXT: cir.store align(8) %[[ASM_RES]], %[[RET]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// LLVM: define{{.*}}@t33
// LLVM: %[[PTR:.*]] = alloca ptr
// Classic codegen doesn't alloca for a return slot.
// CIRLLVMONLY: alloca ptr
// LLVM: %[[RET:.*]] = alloca ptr
// LLVM: %[[PTR_LOAD:.*]] = load ptr, ptr %[[PTR]]
// LLVM: %[[ASM_RES:.*]] = call ptr asm "lea $1, $0", "=r,p,~{dirflag},~{fpsr},~{flags}"(ptr %[[PTR_LOAD]])
// LLVM: store ptr %[[ASM_RES]], ptr %[[RET]]
void *t33(void *ptr)
{
  void *ret;
  asm ("lea %1, %0" : "=r" (ret) : "p" (ptr));
  return ret;
}

//      CIR: %[[FOUR:.*]] = cir.const #cir.int<4> : !s32i
//      CIR: cir.asm(x86_att, 
// CIR-NEXT:   out = [],
// CIR-NEXT:   in = [%[[FOUR]] : !s32i],
// CIR-NEXT:   in_out = [],
// CIR-NEXT:   {"T34 CC NAMED MODIFIER: ${0:c}" "i,~{dirflag},~{fpsr},~{flags}"}) side_effects
// LLVM: define{{.*}}@t34
// LLVM: call void asm sideeffect "T34 CC NAMED MODIFIER: ${0:c}", "i,~{dirflag},~{fpsr},~{flags}"(i32 4)
void t34(void) {
  __asm__ volatile("T34 CC NAMED MODIFIER: %cc[input]" :: [input] "i" (4));
}
