// REQUIRES: hexagon-registered-target
// RUN: %clang++ --target=hexagon-unknown-elf -mv79 -mhvx %s -O2 -fenable-ripple -E -o - | FileCheck %s
// RUN: %clang++ --target=hexagon-unknown-elf -mv79 -mhvx -x c %s -O2 -fenable-ripple -E -o - -D__hexagon__=1 | FileCheck %s
#include <ripple.h>
#include <ripple_hvx.h>

// CHECK: static int32_t hvx_to_ripple_v32i32( ripple_block_t BS, v32i32 x) { int32_t tmp[32]; *((v32i32 *)tmp) = x; return tmp[__builtin_ripple_get_index((BS), (0))]; }
// CHECK: static int32_t hvx_to_ripple_2d_v32i32( ripple_block_t BS, v32i32 x) { int32_t tmp[32]; *((v32i32 *)tmp) = x; return tmp[__builtin_ripple_get_index((BS), (1)) * __builtin_ripple_get_size((BS), (0)) + __builtin_ripple_get_index((BS), (0))]; }

#define N 32
typedef v32i32 v_t;

void g(v_t x) {
  ripple_block_t BS = ripple_set_block_shape(0, N / 2, 2);
  int idx = hvx_to_ripple_2d(BS, 32, i32, x);
  // CHECK: int idx = hvx_to_ripple_2d_v32i32((BS), (x));
}

int main() {
  int x[N];
  for (int i = 0; i != N; ++i)
    x[i] = i;
  g(*(v_t *) x);
}
