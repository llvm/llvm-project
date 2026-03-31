// REQUIRES: target=hexagon{{.*}}, has-ripple-hexagon-rtlib
// RUN: %clang -ffreestanding -Os -S -fenable-ripple -mv75 -mhvx  %s -o - 2>&1 | FileCheck %s
// Simply sanity check that we did not crash and vectorize compares
// CHECK: vcmpw.gtu

// Remove xfail once the upstream bug is fixed
// XFAIL: *

#include <ripple.h>

void nkctv_nkvw_nctw_dram_0(const uint8_t* aptr, const uint8_t* bptr, uint8_t* cptr, size_t T, size_t W) {

  ripple_block_t BS = ripple_set_block_shape((0), 8, 8);
  size_t v0 = ripple_id(BS, (0)), v1 = ripple_id(BS, (1));
  size_t nv0 = ripple_get_block_size(BS, (0)), nv1 = ripple_get_block_size(BS, (1));


  size_t t;
  for (t = 0; t + nv1 <= T; t+=nv1) {
    size_t w;
    for (w = 0; w + nv0 <= W; w+=nv0) {
      cptr[(w+v0) + ((t+v1) )] = (aptr[t+v1] * bptr[w+v0]);
    }
  }
  if (t + v1 < T) {
    size_t w;
    for (w = 0; w + nv0 <= W; w+=nv0) {
      cptr[(w+v0) + ((t+v1) )] = (aptr[t+v1] * bptr[w+v0]);
    }
    if (w + v0 < W) {
      cptr[(w+v0) + ((t+v1) )] = (aptr[t+v1] * bptr[w+v0]);
    }
  }
}
