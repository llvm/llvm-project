// RUN: %clang_cc1 -triple s390x-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

void test() {
  unsigned long sum_high = 1, sum_low = 2;
  register unsigned long a_high asm("r8") = 3;
  register unsigned long a_low asm("r9") = 4;
  unsigned long b_high = 5, b_low = 6;

  __asm__ (
      "algr\t%1,%5\n\t"
      "alcgr\t%0,%3"
      : "=r"(sum_high), "=&r"(sum_low)
      : "0"(a_high), "r"(b_high),
        "%1"(a_low), "r"(b_low)
      : "cc");
}

// CHECK-NOT: "=r,=&r,{r8},r,{r9},r,~{cc}"
// CHECK: "=r,=&r,0,r,%1,r,~{cc}"
