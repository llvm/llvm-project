// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -O2 -emit-llvm %s -o - | FileCheck %s

unsigned long foo(unsigned long addr, unsigned long a0,
                  unsigned long a1, unsigned long a2,
                  unsigned long a3, unsigned long a4,
                  unsigned long a5) {
  register unsigned long result asm("rax");
  register unsigned long addr1 asm("rax") = addr;
  register unsigned long b0 asm("rdi") = a0;
  register unsigned long b1 asm("rsi") = a1;
  register unsigned long b2 asm("rdx") = a2;
  register unsigned long b3 asm("rcx") = a3;
  register unsigned long b4 asm("r8") = a4;
  register unsigned long b5 asm("r9") = a5;

  // CHECK: tail call i64 asm "call *$1", "={rax},{rax},{rdi},{rsi},{rdx},{rcx},{r8},{r9},{rax},~{dirflag},~{fpsr},~{flags}"(i64 %addr, i64 %a0, i64 %a1, i64 %a2, i64 %a3, i64 %a4, i64 %a5, i64 undef)
  asm("call *%1" 
      : "+r" (result) 
      : "r"(addr1), "r"(b0), "r"(b1), "r"(b2), "r"(b3), "r"(b4), "r"(b5));
  return result;
}

unsigned long foo1(unsigned long addr, unsigned long a0,
                  unsigned long a1, unsigned long a2,
                  unsigned long a3, unsigned long a4,
                  unsigned long a5) {
  unsigned long result;
  unsigned long addr1 = addr;
  unsigned long b0 = a0;
  unsigned long b1 = a1;
  unsigned long b2 = a2;
  unsigned long b3 = a3;
  unsigned long b4 = a4;
  unsigned long b5 = a5;

  // CHECK: tail call i64 asm "call *$1", "={rax},{rax},{rdi},{rsi},{rdx},{rcx},{r8},{r9},{rax},~{dirflag},~{fpsr},~{flags}"(i64 %addr, i64 %a0, i64 %a1, i64 %a2, i64 %a3, i64 %a4, i64 %a5, i64 undef)
  asm("call *%1" 
      : "+{rax}" (result) 
      : "{rax}"(addr1), "{rdi}"(b0), "{rsi}"(b1), "{rdx}"(b2), "{rcx}"(b3), "{r8}"(b4), "{r9}"(b5));
  return result;
}

unsigned long foo2(unsigned long addr, unsigned long a0,
                  unsigned long a1, unsigned long a2,
                  unsigned long a3, unsigned long a4,
                  unsigned long a5) {
  register unsigned long result asm("rax");
  unsigned long addr1 = addr;
  unsigned long b0 = a0;
  register unsigned long b1 asm ("rsi") = a1;
  unsigned long b2 = a2;
  unsigned long b3 = a3;
  register unsigned long b4 asm ("r8") = a4;
  unsigned long b5 = a5;

  // CHECK: tail call i64 asm "call *$1", "={rax},{rax},{rdi},{rsi},{rdx},{rcx},{r8},{r9},{rax},~{dirflag},~{fpsr},~{flags}"(i64 %addr, i64 %a0, i64 %a1, i64 %a2, i64 %a3, i64 %a4, i64 %a5, i64 undef)
  asm("call *%1" 
      : "+r" (result) 
      : "{rax}"(addr1), "{rdi}"(b0), "r"(b1), "{rdx}"(b2), "{rcx}"(b3), "r"(b4), "{r9}"(b5));
  return result;
}
