// REQUIRES: hexagon-registered-target
// RUN: %clang -Xclang -disable-llvm-passes --target=hexagon-unknown-elf -mcpu=hexagonv81 -S -emit-llvm %s -o - 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"
// RUN: %clang -x c -Xclang -disable-llvm-passes --target=hexagon-unknown-elf -mcpu=hexagonv81 -S -emit-llvm %s -o - 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"

#include <ripple.h>

#ifdef __cplusplus
extern "C"
#endif
size_t get_src_id(size_t id, size_t n) {
  return (id % 2) ? (id - 1) : (id + 1);
}

#define gen_shuffle_fptr_test(N, LONGTYPE)                                     \
  void check_shuffle_fptr_##LONGTYPE(LONGTYPE in[N], LONGTYPE out[N]) {        \
    ripple_block_t BS = ripple_set_block_shape(0, N);                          \
    int id = ripple_id(BS, 0);                                                 \
    LONGTYPE tmp = in[id];                                                     \
    LONGTYPE shuf_tmp = ripple_shuffle(tmp, get_src_id);                       \
    out[id] = shuf_tmp;                                                        \
  }                                                                            \
  void check_shuffle_pair_fptr_##LONGTYPE(LONGTYPE in[N], LONGTYPE in2[N],     \
                                          LONGTYPE out[N]) {                   \
    ripple_block_t BS = ripple_set_block_shape(0, N);                          \
    int id = ripple_id(BS, 0);                                                 \
    LONGTYPE tmp = in[id];                                                     \
    LONGTYPE tmp2 = in2[id];                                                   \
    LONGTYPE shuf_tmp = ripple_shuffle_pair(tmp, tmp2, get_src_id);            \
    out[id] = shuf_tmp;                                                        \
  }

gen_shuffle_fptr_test(16, _Float16);
// CHECK: check_shuffle_fptr__Float16
// CHECK: llvm.ripple.shuffle.f16(half %{{[0-9]+}}, half %{{[0-9]+}}, i1 false, ptr
// CHECK: ret
// CHECK: check_shuffle_pair_fptr__Float16
// CHECK: llvm.ripple.shuffle.f16(half %{{[0-9]+}}, half %{{[0-9]+}}, i1 true, ptr
// CHECK: ret

// gen_shuffle_fptr_test(16, __bf16);
// CHECK-TODO: check_shuffle_fptr___bf16
// CHECK-TODO: llvm.ripple.shuffle.f32(float %{{.*}}, float %{{.*}}, i1 false, ptr
// CHECK-TODO: ret
// CHECK-TODO: check_shuffle_pair_fptr___bf16
// CHECK-TODO: llvm.ripple.shuffle.f32(float %{{.*}}, float %{{.*}}, i1 true, ptr
// CHECK-TODO: ret
