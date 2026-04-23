// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm -O2 %s -o - | FileCheck %s

void test_reg_mem_inputs(unsigned long flags) {
  // CHECK-LABEL: @test_reg_mem_inputs
  // CHECK:         callbr void @llvm.asm.constraint.br()
  // CHECK-NEXT:            to label %asm.pref.reg [label %asm.pref.mem]
  // CHECK:       asm.pref.reg:
  // CHECK-NEXT:     tail call void asm sideeffect "", "rm,~{dirflag},~{fpsr},~{flags}"(i32 %flags)
  // CHECK-NEXT:     br label %asm.merge
  // CHECK:       asm.pref.mem:
  // CHECK-NEXT:     tail call void asm sideeffect "", "rm,~{dirflag},~{fpsr},~{flags}"(i32 %flags)
  // CHECK-NEXT:     br label %asm.merge
  asm ("" : : "rm" (flags));
}

unsigned long test_reg_mem_outputs(void) {
  // CHECK-LABEL: @test_reg_mem_outputs
  // CHECK:         callbr void @llvm.asm.constraint.br()
  // CHECK-NEXT:            to label %asm.pref.reg [label %asm.pref.mem]
  // CHECK:       asm.pref.reg:
  // CHECK-NEXT:    = tail call i32 asm "", "=rm,~{dirflag},~{fpsr},~{flags}"()
  // CHECK-NEXT:    br label %asm.merge
  // CHECK:       asm.pref.mem:
  // CHECK-NEXT:    call void asm "", "=*rm,~{dirflag},~{fpsr},~{flags}"(ptr nonnull elementtype(i32) %out)
  // CHECK:         = load i32, ptr %out
  // CHECK-NEXT:    br label %asm.merge
  unsigned long out;
  asm ("" : "=rm" (out));
  return out;
}

void test_g_inputs(unsigned long flags) {
  // CHECK-LABEL: @test_g_inputs
  // CHECK:         callbr void @llvm.asm.constraint.br()
  // CHECK-NEXT:            to label %asm.pref.reg [label %asm.pref.mem]
  // CHECK:       asm.pref.reg:
  // CHECK-NEXT:    tail call void asm sideeffect "", "imr,~{dirflag},~{fpsr},~{flags}"(i32 %flags)
  // CHECK-NEXT:    br label %asm.merge
  // CHECK:       asm.pref.mem:
  // CHECK-NEXT:    tail call void asm sideeffect "", "imr,~{dirflag},~{fpsr},~{flags}"(i32 %flags)
  // CHECK-NEXT:    br label %asm.merge
  asm ("" : : "g" (flags));
}

unsigned long test_g_outputs(void) {
  // CHECK-LABEL: @test_g_outputs
  // CHECK:         callbr void @llvm.asm.constraint.br()
  // CHECK-NEXT:            to label %asm.pref.reg [label %asm.pref.mem]
  // CHECK:       asm.pref.reg:
  // CHECK-NEXT:    %0 = tail call i32 asm "", "=imr,~{dirflag},~{fpsr},~{flags}"()
  // CHECK-NEXT:    br label %asm.merge
  // CHECK:       asm.pref.mem:
  // CHECK-NEXT:    call void asm "", "=*imr,~{dirflag},~{fpsr},~{flags}"(ptr nonnull elementtype(i32) %out)
  // CHECK-NEXT:    = load i32, ptr %out
  // CHECK-NEXT:    br label %asm.merge
  unsigned long out;
  asm ("" : "=g" (out));
  return out;
}

void test_reg_mem_earlyclobber(int len) {
  // CHECK-LABEL: @test_reg_mem_earlyclobber
  // CHECK:         callbr void @llvm.asm.constraint.br()
  // CHECK-NEXT:            to label %asm.pref.reg [label %asm.pref.mem]
  // CHECK:       asm.pref.reg:
  // CHECK-NEXT:    = tail call i32 asm sideeffect "", "=&rm,0,~{dirflag},~{fpsr},~{flags}"(i32 %len)
  // CHECK-NEXT:    br label %asm.merge
  // CHECK:       asm.pref.mem:
  // CHECK-NEXT:    call void asm sideeffect "", "=*&rm,0,~{dirflag},~{fpsr},~{flags}"(ptr nonnull elementtype(i32) %len.addr, i32 %len)
  // CHECK-NEXT:    br label %asm.merge
  __asm__ volatile ("" : "+&&rm" (len));
}

void test_reg_mem_commutative(int len) {
  // CHECK-LABEL: @test_reg_mem_commutative
  // CHECK:         callbr void @llvm.asm.constraint.br()
  // CHECK-NEXT:            to label %asm.pref.reg [label %asm.pref.mem]
  // CHECK:       asm.pref.reg:
  // CHECK-NEXT:    = tail call { i32, i32 } asm sideeffect "", "=%rm,=rm,0,1,~{dirflag},~{fpsr},~{flags}"(i32 %len, i32 %len)
  // CHECK-NEXT:    br label %asm.merge
  // CHECK:       asm.pref.mem:
  // CHECK-NEXT:    call void asm sideeffect "", "=*%rm,=*rm,0,1,~{dirflag},~{fpsr},~{flags}"(ptr nonnull elementtype(i32) %len.addr, ptr nonnull elementtype(i32) %len.addr, i32 %len, i32 %len)
  // CHECK-NEXT:    br label %asm.merge
  __asm__ volatile ("" : "+%%rm" (len), "+rm" (len));
}

unsigned long test_asm_goto(void) {
  // CHECK-LABEL: @test_asm_goto
  // CHECK:         callbr void @llvm.asm.constraint.br()
  // CHECK-NEXT:         to label %asm.pref.reg [label %asm.pref.mem]
  // CHECK:       asm.pref.reg:
  // CHECK-NEXT:    = callbr i32 asm "", "=rm,!i,~{dirflag},~{fpsr},~{flags}"()
  // CHECK-NEXT:         to label %cleanup [label %indirect.split]
  // CHECK:       asm.pref.mem:
  // CHECK-NEXT:    callbr void asm "", "=*rm,!i,~{dirflag},~{fpsr},~{flags}"(ptr nonnull elementtype(i32) %out)
  // CHECK-NEXT:         to label %asm.pref.mem.asm.merge_crit_edge [label %cleanup]
  // CHECK:       asm.pref.mem.asm.merge_crit_edge:
  // CHECK-NEXT:    = load i32, ptr %out, align 4, !tbaa !8
  // CHECK-NEXT:    br label %cleanup
  // CHECK:       indirect.split:
  // CHECK-NEXT:    br label %cleanup
  unsigned long out;
  asm goto ("" : "=rm" (out) ::: indirect);
  return out;

indirect:
  return 42;
}

// PR3908
void test_pr3908(int r) {
  // CHECK-LABEL: @test_pr3908
  // CHECK:         callbr void @llvm.asm.constraint.br()
  // CHECK-NEXT:            to label %asm.pref.reg [label %asm.pref.mem]
  // CHECK:       asm.pref.reg:
  // CHECK-NEXT:    = tail call i32 asm "# PR3908 $1 $3 $2 $0", "=r,mx,mr,x,0,~{dirflag},~{fpsr},~{flags}"(i32 0, i32 0, double 0.000000e+00, i32 %r)
  // CHECK-NEXT:    br label %asm.merge
  // CHECK:       asm.pref.mem:                                     ; preds = %entry
  // CHECK-NEXT:    = tail call i32 asm "# PR3908 $1 $3 $2 $0", "=r,mx,mr,x,0,~{dirflag},~{fpsr},~{flags}"(i32 0, i32 0, double 0.000000e+00, i32 %r)
  // CHECK-NEXT:    br label %asm.merge
  __asm__ ("# PR3908 %[lf] %[xx] %[li] %[r]"
           : [r] "+r" (r)
           : [lf] "mx" (0), [li] "mr" (0), [xx] "x" ((double)(0)));
}
