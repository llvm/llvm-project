// RUN: %clang -ffreestanding -g -fenable-ripple -O2 %s -S -emit-llvm -o %t 2> %t.err; FileCheck %s --input-file=%t.err
// This checks that we don't delete a PHI outside the if-conversion zone (we still track it's shape)
#include "ripple_test.h"


// CHECK:      unsupported-loop.c:22:11: error: unsupported vectorization of vector branch when it applies to a non single-entry-single-exit (SESE) region or simple vector loops (one exit)
// CHECK-NEXT:   22 |       if (!ripple_reduceor(0b1, is_not_diverging))
// CHECK-NEXT:      |           ^

void __attribute__((noinline))
check(const float *x0, const float y0, const int width, const int height,
                  const int max_iters, int *output) {
  ripple_block_t BS = ripple_set_block_shape(0, 8);

  ripple_parallel_full(BS, 0);
  for (int i = 0; i < width; ++i) {
    float z_re = x0[i], z_im = ripple_broadcast(BS, 0b1, y0);
    int i_at_break = ripple_broadcast(BS, 0b1, 0);
    for (int i = 0; i < max_iters; ++i) {
      int8_t is_not_diverging = z_re + z_im;

      if (!ripple_reduceor(0b1, is_not_diverging))
        break;

      if (is_not_diverging) {
        i_at_break += 1;
        z_re *= 34.f - z_im;
        z_im *= 499.f * z_re;
      }
    }
    output[i] = i_at_break;
  }
}
