// RUN: %clang -mmmx -target i386-unknown-unknown -emit-llvm -S %s -o - | FileCheck %s
// <rdar://problem/9091220>
#include <mmintrin.h>

// CHECK: { x86_mmx, x86_mmx, x86_mmx, x86_mmx, x86_mmx, x86_mmx, x86_mmx }

void foo(long long fill) {
  __m64 vfill = _mm_cvtsi64_m64(fill);
  __m64 v1, v2, v3, v4, v5, v6, v7;

  __asm__ __volatile__ (
    "\tmovq  %7, %0\n"
    "\tmovq  %7, %1\n"
    "\tmovq  %7, %2\n"
    "\tmovq  %7, %3\n"
    "\tmovq  %7, %4\n"
    "\tmovq  %7, %5\n"
    "\tmovq  %7, %6"
    : "=&y" (v1), "=&y" (v2), "=&y" (v3),
      "=&y" (v4), "=&y" (v5), "=&y" (v6), "=y" (v7)
    : "y" (vfill));
}
