// RUN: %clang_cc1 -triple nvptx-unknown-unknown -O3 -S -o - %s -emit-llvm | FileCheck %s
// RUN: %clang_cc1 -triple nvptx64-unknown-unknown -O3 -S -o - %s -emit-llvm | FileCheck %s

void constraints(void) {
  char           c;
  unsigned char  uc;
  short          s;
  unsigned short us;
  int            i;
  unsigned int   ui;
  long long      ll;
  unsigned long long ull;
  float          f;
  double         d;

  // CHECK: i8 asm sideeffect "mov.b8 $0, $1;", "=c,c"
  asm volatile ("mov.b8 %0, %1;" : "=c"(c) : "c"(c));
  // CHECK: i8 asm sideeffect "mov.b8 $0, $1;", "=c,c"
  asm volatile ("mov.b8 %0, %1;" : "=c"(uc) : "c"(uc));

  // CHECK: i16 asm sideeffect "mov.b16 $0, $1;", "=h,h"
  asm volatile ("mov.b16 %0, %1;" : "=h"(s) : "h"(s));
  // CHECK: i16 asm sideeffect "mov.b16 $0, $1;", "=h,h"
  asm volatile ("mov.b16 %0, %1;" : "=h"(us) : "h"(us));

  // CHECK: i32 asm sideeffect "mov.b32 $0, $1;", "=r,r"
  asm volatile ("mov.b32 %0, %1;" : "=r"(i) : "r"(i));
  // CHECK: i32 asm sideeffect "mov.b32 $0, $1;", "=r,r"
  asm volatile ("mov.b32 %0, %1;" : "=r"(ui) : "r"(ui));

  // CHECK: i64 asm sideeffect "mov.b64 $0, $1;", "=l,l"
  asm volatile ("mov.b64 %0, %1;" : "=l"(ll) : "l"(ll));
  // CHECK: i64 asm sideeffect "mov.b64 $0, $1;", "=l,l"
  asm volatile ("mov.b64 %0, %1;" : "=l"(ull) : "l"(ull));

  // CHECK: float asm sideeffect "mov.b32 $0, $1;", "=f,f"
  asm volatile ("mov.b32 %0, %1;" : "=f"(f) : "f"(f));
  // CHECK: double asm sideeffect "mov.b64 $0, $1;", "=d,d"
  asm volatile ("mov.b64 %0, %1;" : "=d"(d) : "d"(d));
}
