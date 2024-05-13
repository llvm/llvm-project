// RUN: %clang_cc1 -emit-llvm %s -o - -triple powerpc64le-linux-gnu | FileCheck %s
// Sadly since this requires a register constraint to trigger we have to set
// a target here.
void a(void) {
  register unsigned long __sc_0 __asm__("r0");
  __asm__ __volatile__("mfcr %0" : "=&r"(__sc_0) : "0"(__sc_0));
}

// Check that we can generate code for this correctly. The matching input
// constraint should not have an early clobber on it.
// CHECK: call i64 asm sideeffect "mfcr $0", "=&{r0},{r0}"
