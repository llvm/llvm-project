// RUN: %clang_cc1 -triple arm64-apple-ios -emit-llvm -o - %s | FileCheck %s

// rdar://9167275

int t1(void)
{
  int x;
  __asm__("mov %0, 7" : "=r" (x));
  return x;
}

long t2(void)
{
  long x;
  __asm__("mov %0, 7" : "=r" (x));
  return x;
}

long t3(void)
{
  long x;
  __asm__("mov %w0, 7" : "=r" (x));
  return x;
}

// rdar://9281206

void t4(long op) {
  long x1;
  asm ("mov x0, %1; svc #0;" : "=r"(x1) :"r"(op),"r"(x1) :"x0" );
}

// rdar://9394290

float t5(float x) {
  __asm__("fadd %0, %0, %0" : "+w" (x));
  return x;
}

// rdar://9865712
void t6 (void *f, int g) {
  // CHECK: t6
  // CHECK: call void asm "str $1, $0", "=*Q,r"
  asm("str %1, %0" : "=Q"(f) : "r"(g));
}
