// RUN: %clang_cc1 %s -triple riscv32 -verify -fsyntax-only
// RUN: %clang_cc1 %s -triple riscv64 -verify -fsyntax-only

void i (void) {
  asm volatile ("" ::: "x0",  "x1",  "x2",  "x3",  "x4",  "x5",  "x6",  "x7");
  asm volatile ("" ::: "x8",  "x9",  "x10", "x11", "x12", "x13", "x14", "x15");
  asm volatile ("" ::: "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23");
  asm volatile ("" ::: "x24", "x25", "x26", "x27", "x28", "x29", "x30", "x31");

  asm volatile ("" ::: "zero", "ra", "sp",  "gp",  "tp", "t0", "t1", "t2");
  asm volatile ("" ::: "s0",   "s1", "a0",  "a1",  "a2", "a3", "a4", "a5");
  asm volatile ("" ::: "a6",   "a7", "s2",  "s3",  "s4", "s5", "s6", "s7");
  asm volatile ("" ::: "s8",   "s9", "s10", "s11", "t3", "t4", "t5", "t6");
}

void f (void) {
  asm volatile ("" ::: "f0",  "f1",  "f2",  "f3",  "f4",  "f5",  "f6",  "f7");
  asm volatile ("" ::: "f8",  "f9",  "f10", "f11", "f12", "f13", "f14", "f15");
  asm volatile ("" ::: "f16", "f17", "f18", "f19", "f20", "f21", "f22", "f23");
  asm volatile ("" ::: "f24", "f25", "f26", "f27", "f28", "f29", "f30", "f31");

  asm volatile ("" ::: "ft0", "ft1", "ft2",  "ft3",  "ft4", "ft5", "ft6",  "ft7");
  asm volatile ("" ::: "fs0", "fs1", "fa0",  "fa1",  "fa2", "fa3", "fa4",  "fa5");
  asm volatile ("" ::: "fa6", "fa7", "fs2",  "fs3",  "fs4", "fs5", "fs6",  "fs7");
  asm volatile ("" ::: "fs8", "fs9", "fs10", "fs11", "ft8", "ft9", "ft10", "ft11");
}

register char i1 __asm__ ("x1"); // expected-error {{size of register 'x1' does not match variable size}}
#if __riscv_xlen == 32
register long long ll2 __asm__ ("x2"); // expected-error {{size of register 'x2' does not match variable size}}
register int i2 __asm__ ("x3");
#endif
register long l3 __asm__ ("x4");
register long ra __asm__ ("ra");
register long sp __asm__ ("sp");
register int *gp __asm__ ("gp");
register char *tp __asm__ ("tp");
register long a7 __asm__ ("a7");
register long s11 __asm__ ("s11");
register long t5 __asm__ ("t5");
register long* f1 __asm__ ("f1"); // expected-error {{register 'f1' unsuitable for global register variables on this target}}
