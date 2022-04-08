// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -no-opaque-pointers -triple armv7-apple-darwin9 -emit-llvm -w -o - %s | FileCheck %s

typedef long long int64_t;
typedef unsigned int uint32_t;

int64_t foo(int64_t v, volatile int64_t *p)
{
  register uint32_t rl asm("r1");
  register uint32_t rh asm("r2");

  int64_t r;
  uint32_t t;

  __asm__ __volatile__(							\
		       "ldrexd%[_rl], %[_rh], [%[_p]]"			\
		       : [_rl] "=&r" (rl), [_rh] "=&r" (rh)		\
		       : [_p] "p" (p) : "memory");

  // CHECK: call { i32, i32 } asm sideeffect "ldrexd$0, $1, [$2]", "=&{r1},=&{r2},r,~{memory}"(i64*

  return r;
}

// Make sure we translate register names properly.
void bar (void) {
  register unsigned int rn asm("r14");
  register unsigned int d asm("r2");

  // CHECK: call i32 asm sideeffect "sub $1, $1, #32", "={r2},{lr}"
  asm volatile ("sub %1, %1, #32" : "=r"(d) : "r"(rn));
}
