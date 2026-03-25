//==============================================================================
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Part of the Ripple vector library to support vector splicing for HVX.
//
//==============================================================================

#include <cstddef>
#include <hexagon_protos.h>
#include <ripple_hvx.h>

// _______________________________ splice / lsplice ______________________________

extern "C" {

#define __decl_hvx_splice(N, T, SHIFT) \
  __attribute__((used, always_inline, weak)) v##N##T \
  ripple_pure_hvx_splice_##T(v##N##T lo, v##N##T hi, size_t start) { \
    return Q6_V_valign_VVR(hi, lo, (start) << SHIFT); \
  }

#define __decl_hvx_lsplice(N, T, SHIFT) \
  __attribute__((used, always_inline, weak)) v##N##T \
  ripple_pure_hvx_lsplice_##T(v##N##T lo, v##N##T hi, size_t start) { \
    return Q6_V_vlalign_VVR(hi, lo, (start) << SHIFT); \
  }

__decl_hvx_splice(128, i8, 0);
__decl_hvx_splice(64, i16, 1);
__decl_hvx_splice(32, i32, 2);
__decl_hvx_splice(16, i64, 3);

__decl_hvx_splice(128, u8, 0);
__decl_hvx_splice(64, u16, 1);
__decl_hvx_splice(32, u32, 2);
__decl_hvx_splice(16, u64, 3);

__decl_hvx_splice(16, f64, 3);
__decl_hvx_splice(32, f32, 2);
#if __has_bf16__
__decl_hvx_splice(64, bf16, 1);
#endif
#if __has_Float16__
__decl_hvx_splice(64, f16, 1);
#endif

__decl_hvx_lsplice(128, i8, 0);
__decl_hvx_lsplice(64, i16, 1);
__decl_hvx_lsplice(32, i32, 2);
__decl_hvx_lsplice(16, i64, 3);

__decl_hvx_lsplice(128, u8, 0);
__decl_hvx_lsplice(64, u16, 1);
__decl_hvx_lsplice(32, u32, 2);
__decl_hvx_lsplice(16, u64, 3);

__decl_hvx_lsplice(16, f64, 3);
__decl_hvx_lsplice(32, f32, 2);
#if __has_bf16__
__decl_hvx_lsplice(64, bf16, 1);
#endif
#if __has_Float16__
__decl_hvx_lsplice(64, f16, 1);
#endif

#undef __decl_hvx_splice
#undef __decl_hvx_lsplice
}