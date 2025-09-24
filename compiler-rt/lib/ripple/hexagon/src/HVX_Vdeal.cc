//==============================================================================
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==============================================================================
//
// Part of the Ripple vector library to support the HVX vector deal and shuffle
// instructions.
//
//==============================================================================
//

#include <hexagon_protos.h>
#include <ripple_hvx.h>

// _______________________________ vdeal / vshuff ______________________________

extern "C" {

#define __decl_hvx_vdeal(N_ELEM, T)                                            \
  __attribute__((used, always_inline, weak)) v##N_ELEM##T                      \
  ripple_pure_hvx_vdeal_##T(v##N_ELEM##T src, size_t chunk_size) {             \
    auto left = Q6_V_lo_W(src);                                                \
    auto right = Q6_V_hi_W(src);                                               \
    return Q6_W_vdeal_VVR(right, left, chunk_size);                            \
  }

#define __decl_hvx_vshuff(N_ELEM, T)                                           \
  __attribute__((used, always_inline, weak)) v##N_ELEM##T                      \
  ripple_pure_hvx_vshuff_##T(v##N_ELEM##T src, size_t chunk_size) {            \
    auto left = Q6_V_lo_W(src);                                                \
    auto right = Q6_V_hi_W(src);                                               \
    return Q6_W_vshuff_VVR(right, left, chunk_size);                           \
  }

__decl_hvx_vdeal(256, i8);
__decl_hvx_vdeal(128, i16);
__decl_hvx_vdeal(64, i32);
__decl_hvx_vdeal(32, i64);

__decl_hvx_vdeal(256, u8);
__decl_hvx_vdeal(128, u16);
__decl_hvx_vdeal(64, u32);
__decl_hvx_vdeal(32, u64);

__decl_hvx_vdeal(32, f64);
__decl_hvx_vdeal(64, f32);
#if __has_bf16__
__decl_hvx_vdeal(128, bf16);
#endif
#if __has_Float16__
__decl_hvx_vdeal(128, f16);
#endif

__decl_hvx_vshuff(256, i8);
__decl_hvx_vshuff(128, i16);
__decl_hvx_vshuff(64, i32);
__decl_hvx_vshuff(32, i64);

__decl_hvx_vshuff(256, u8);
__decl_hvx_vshuff(128, u16);
__decl_hvx_vshuff(64, u32);
__decl_hvx_vshuff(32, u64);

__decl_hvx_vshuff(32, f64);
__decl_hvx_vshuff(64, f32);
#if __has_bf16__
__decl_hvx_vshuff(128, bf16);
#endif
#if __has_Float16__
__decl_hvx_vshuff(128, f16);
#endif

#undef __decl_hvx_vdeal
#undef __decl_hvx_vshuff
}
