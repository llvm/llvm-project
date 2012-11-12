// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s

unsigned long long foo(unsigned long long addr, unsigned long long a0,
                       unsigned long long a1, unsigned long long a2,
                       unsigned long long a3, unsigned long long a4,
                       unsigned long long a5) {
  register unsigned long long result asm("rax");
  register unsigned long long b0 asm("rdi");
  register unsigned long long b1 asm("rsi");
  register unsigned long long b2 asm("rdx");
  register unsigned long long b3 asm("rcx");
  register unsigned long long b4 asm("r8");
  register unsigned long long b5 asm("r9");

  b0 = a0;
  b1 = a1;
  b2 = a2;
  b3 = a3;
  b4 = a4;
  b5 = a5;

  asm("call *%1" : "=r" (result)
      : "r"(addr), "r" (b0), "r" (b1), "r" (b2), "r" (b3), "r" (b4), "r" (b5));
  return result;
}

// CHECK: call i64 asm "call *$1", "={rax},r,{rdi},{rsi},{rdx},{rcx},{r8},{r9},~{dirflag},~{fpsr},~{flags}"

unsigned long long foo2(unsigned long long addr, double a0,
                       double a1, double a2,
                       double a3, double a4,
                       double a5, double a6, double a7) {
  register double b0 asm("xmm0");
  register double b1 asm("xmm1");
  register double b2 asm("xmm2");
  register double b3 asm("xmm3");
  register double b4 asm("xmm4");
  register double b5 asm("xmm5");
  register double b6 asm("xmm6");
  register double b7 asm("xmm7");

  register unsigned long long result asm("rax");

  b0 = a0;
  b1 = a1;
  b2 = a2;
  b3 = a3;
  b4 = a4;
  b5 = a5;
  b6 = a6;
  b7 = a7;

  asm("call *%1" : "=r" (result)
      : "r"(addr), "x" (b0), "x" (b1), "x" (b2), "x" (b3), "x" (b4), "x" (b5), "x" (b6),
        "x" (b7));
  return result;
}

// CHECK: call i64 asm "call *$1", "={rax},r,{xmm0},{xmm1},{xmm2},{xmm3},{xmm4},{xmm5},{xmm6},{xmm7},~{dirflag},~{fpsr},~{flags}

int randomvar asm("randomvar");
void foo3() {
  asm("vartest %0" : : "r"(randomvar));
}
// CHECK: call void asm sideeffect "vartest $0", "r,~{dirflag},~{fpsr},~{flags}"
