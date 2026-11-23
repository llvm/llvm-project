// RUN: %clang_cc1 -verify -ffreestanding -triple=aarch64-linux-gnu %s
// REQUIRES: aarch64-registered-target

#include <stdatomic.h>

void memory_checks(_Float16 *p16, __bf16 *pbf, float *pf, double *pd) {
  (void)__atomic_fetch_min(p16, (_Float16)1.0f, memory_order_relaxed);
  (void)__atomic_fetch_max(pbf, (__bf16)2.0f, memory_order_acquire);
  (void)__atomic_fetch_min(pf, 3.0f, memory_order_release);
  (void)__atomic_fetch_max(pd, 4.0, memory_order_seq_cst);
}

void nullPointerWarning(void) {
  (void)__atomic_fetch_min((volatile float*)0, 42.0, memory_order_relaxed); // expected-warning {{null passed to a callee that requires a non-null argument}}
  (void)__atomic_fetch_max((float*)0, 42.0, memory_order_relaxed); // expected-warning {{null passed to a callee that requires a non-null argument}}
}
