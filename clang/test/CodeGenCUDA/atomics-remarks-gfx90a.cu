// RUN: %clang_cc1 %s -triple=amdgcn-amd-amdhsa -fcuda-is-device \
// RUN:   -target-cpu gfx90a -Rpass=atomic-expand -S -o - 2>&1 | \
// RUN:   FileCheck %s --check-prefix=GFX90A-CAS

// REQUIRES: amdgpu-registered-target

#include "Inputs/cuda.h"
#include <stdatomic.h>

// GFX90A-CAS: A compare and swap loop was generated for an atomic fadd operation at system memory scope
// GFX90A-CAS-LABEL: _Z14atomic_add_casPf
// GFX90A-CAS:  flat_atomic_cmpswap
// GFX90A-CAS:  s_cbranch_execnz
__device__ float atomic_add_cas(float *p) {
  return __atomic_fetch_add(p, 1.0f, memory_order_relaxed);
}
