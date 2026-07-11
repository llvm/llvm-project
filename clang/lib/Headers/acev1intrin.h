/*===-------------- acev1intrin.h - ACEV1 intrinsics -*- C/C++ -*------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===------------------------------------------------------------------------===
 */

#ifndef __IMMINTRIN_H
#error "Never use <acev1intrin.h> directly; include <immintrin.h> instead."
#endif /* __IMMINTRIN_H */

#ifndef __ACEV1INTRIN_H
#define __ACEV1INTRIN_H
#ifdef __x86_64__

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS_ACE                                                 \
  __attribute__((__always_inline__, __nodebug__, __target__("acev1")))

/// Vector type for combining two 512-bit halves into 1024-bit BSR value.
typedef int __v32si __attribute__((__vector_size__(128)));

/// Combine two 512-bit vector halves into a single 1024-bit vector.
/// This is an internal helper for BSR intrinsics.
///
/// \param __lo
///    The low 512-bit half (B-scales, BSR bits [511:0]).
/// \param __hi
///    The high 512-bit half (A-scales, BSR bits [1023:512]).
/// \returns A 1024-bit vector with __lo in elements [0:15] and __hi in [16:31].
static __inline__ __v32si __DEFAULT_FN_ATTRS_ACE
__bsr_combine_v32(__v16si __lo, __v16si __hi) {
  return __builtin_shufflevector(__lo, __hi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                 23, 24, 25, 26, 27, 28, 29, 30, 31);
}

/// \struct __bsr
/// \brief BSR (Block Scale Register) struct type.
///
/// The __bsr struct bundles the low and high 512-bit halves of the 1024-bit
/// Block Scale Register into a single value type. This mirrors the __tile1024i
/// pattern from AMX, but without shape metadata (BSR has fixed 1024-bit size).
///
/// BSR layout per ACE spec:
///   - BSR[511:0]    = B-scales (lo) - column/B-input scales
///   - BSR[1023:512] = A-scales (hi) - row/A-input scales
///
/// Usage:
/// \code
///   __bsr scales = __bsr_make(lo_zmm, hi_zmm);
///   __bsr_store(scales);  // write to hardware BSR before compute
/// \endcode
typedef struct __bsr_str {
  __m512i lo; ///< Low 512-bit half (B-scales, BSR bits [511:0])
  __m512i hi; ///< High 512-bit half (A-scales, BSR bits [1023:512])
} __bsr;

/// Construct a BSR value from low and high 512-bit halves.
///
/// \headerfile <immintrin.h>
///
/// \param __lo
///    The low 512-bit half (B-scales, BSR bits [511:0]).
/// \param __hi
///    The high 512-bit half (A-scales, BSR bits [1023:512]).
/// \returns A __bsr struct containing both halves.
static __inline__ __bsr __DEFAULT_FN_ATTRS_ACE __bsr_make(__m512i __lo,
                                                          __m512i __hi) {
  __bsr __b;
  __b.lo = __lo;
  __b.hi = __hi;
  return __b;
}

/// Extract the low 512-bit half (B-scales) from a BSR value.
///
/// \headerfile <immintrin.h>
///
/// \param __b
///    The BSR value to extract from.
/// \returns The low 512-bit half (B-scales, BSR bits [511:0]).
static __inline__ __m512i __DEFAULT_FN_ATTRS_ACE __bsr_get_lo(__bsr __b) {
  return __b.lo;
}

/// Extract the high 512-bit half (A-scales) from a BSR value.
///
/// \headerfile <immintrin.h>
///
/// \param __b
///    The BSR value to extract from.
/// \returns The high 512-bit half (A-scales, BSR bits [1023:512]).
static __inline__ __m512i __DEFAULT_FN_ATTRS_ACE __bsr_get_hi(__bsr __b) {
  return __b.hi;
}

/// Return a new BSR value with the low half (B-scales) replaced.
///
/// \headerfile <immintrin.h>
///
/// \param __b
///    The original BSR value.
/// \param __lo
///    The new low 512-bit half (B-scales).
/// \returns A new __bsr with the low half replaced.
static __inline__ __bsr __DEFAULT_FN_ATTRS_ACE __bsr_set_lo(__bsr __b,
                                                            __m512i __lo) {
  __b.lo = __lo;
  return __b;
}

/// Return a new BSR value with the high half (A-scales) replaced.
///
/// \headerfile <immintrin.h>
///
/// \param __b
///    The original BSR value.
/// \param __hi
///    The new high 512-bit half (A-scales).
/// \returns A new __bsr with the high half replaced.
static __inline__ __bsr __DEFAULT_FN_ATTRS_ACE __bsr_set_hi(__bsr __b,
                                                            __m512i __hi) {
  __b.hi = __hi;
  return __b;
}

/// Store a BSR value to the hardware Block Scale Register.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> BSRMOVF </c> instruction.
///
/// Per ACE spec: BSRMOVF writes A-scales (hi) to BSR[1023:512] and
/// B-scales (lo) to BSR[511:0].
///
/// \param __b
///    The BSR value to store to the hardware register.
static __inline__ void __DEFAULT_FN_ATTRS_ACE __bsr_store(__bsr __b) {
  __builtin_ia32_bsrmovf((__v16si)__b.hi, (__v16si)__b.lo);
}

/// Load the current hardware BSR state into a __bsr struct.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> BSRMOVH </c> and <c> BSRMOVL </c>
/// instructions (read forms).
///
/// \returns A __bsr struct containing the current hardware BSR state.
static __inline__ __bsr __DEFAULT_FN_ATTRS_ACE __bsr_load(void) {
  __bsr __b;
  __b.lo = (__m512i)__builtin_ia32_bsrmovl_get();
  __b.hi = (__m512i)__builtin_ia32_bsrmovh_get();
  return __b;
}

/// Load tile configuration from a 64-byte memory location. For ACE
/// (Palette 2), the palette_id byte must be 2. Unlike AMX (Palette 1),
/// ACE tiles have fixed dimensions of 16 rows × 64 bytes, so per-tile
/// row/column configuration bytes are ignored. If palette_id is zero,
/// tiles return to init state and are zeroed. Invalid configurations
/// result in #GP fault.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> LDTILECFG </c> instruction.
///
/// \param __config
///    A pointer to 64-byte tile configuration (use __ace_tile_config).
static __inline__ void __DEFAULT_FN_ATTRS_ACE
_tile_ace_loadconfig(const void *__config) {
  __builtin_ia32_tile_loadconfig(__config);
}

/// Store the current tile configuration to a 64-byte memory location.
/// For ACE (Palette 2), the stored palette_id will be 2 and per-tile
/// configuration bytes reflect the fixed 16×64 dimensions. If tiles
/// are not configured, all zeroes are stored.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> STTILECFG </c> instruction.
///
/// \param __config
///    A pointer to 64-byte tile configuration buffer.
static __inline__ void __DEFAULT_FN_ATTRS_ACE
_tile_ace_storeconfig(void *__config) {
  __builtin_ia32_tile_storeconfig(__config);
}

/// Release the tile configuration to return to init state, releasing
/// all tile storage. After this, tiles must be reconfigured with
/// _tile_ace_loadconfig before use.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TILERELEASE </c> instruction.
static __inline__ void __DEFAULT_FN_ATTRS_ACE _tile_ace_release(void) {
  __builtin_ia32_tilerelease();
}

/// Zero the ACE tile specified by "tile". Sets all 1024 bytes
/// (16 rows × 64 bytes) of the tile register to zero.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TILEZERO </c> instruction.
///
/// \param tile
///    Destination tile register ID (0-7).
#define _tile_ace_zero(tile) __builtin_ia32_tilezero((tile))

/// Move a 64-byte ZMM vector to a tile column. The 16 doublewords from
/// the ZMM are written as a vertical column in the tile at the specified
/// index.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TILEMOVCOL </c> instruction.
///
/// \param dst
///    Destination tile register ID (0-7).
/// \param src
///    Source ZMM register ID (0-31).
/// \param idx
///    Column index (0-15). Immediate or register form selected automatically.
#define _tile_setcol(dst, src, idx)                                            \
  __builtin_ia32_tilesetcol((dst), (src), (idx))

/// Move a 64-byte ZMM vector to a tile row. The ZMM contents are written
/// as a horizontal row in the tile at the specified index.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TILEMOVROW </c> instruction.
///
/// \param dst
///    Destination tile register ID (0-7).
/// \param src
///    Source ZMM register ID (0-31).
/// \param idx
///    Row index (0-15). Immediate or register form selected automatically.
#define _tile_setrow(dst, src, idx)                                            \
  __builtin_ia32_tilesetrow((dst), (src), (idx))

/// Initialize the Block Scale Register (BSR) to zero. The BSR holds
/// 32 scaling factors used by mixed-precision outer product instructions
/// (TOP4MX* variants). Must be called before using BSR-scaled operations.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> BSRINIT </c> instruction.
static __inline__ void __DEFAULT_FN_ATTRS_ACE _bsr0_init(void) {
  __builtin_ia32_bsrinit();
}

/// Load the full BSR (32 scale factors) from two ZMM registers. The low
/// half (16 factors) comes from __src1, high half from __src2. Each
/// doubleword contains one scale factor.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> BSRMOVF </c> instruction.
///
/// \param __src1
///    ZMM with scale factors for BSR low half (factors 0-15).
/// \param __src2
///    ZMM with scale factors for BSR high half (factors 16-31).
static __inline__ void __DEFAULT_FN_ATTRS_ACE _bsr0_movf(__m512i __src1,
                                                         __m512i __src2) {
  __builtin_ia32_bsrmovf((__v16si)__src1, (__v16si)__src2);
}

/// Load the high half of BSR (scale factors 16-31) from a ZMM register.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> BSRMOVH </c> instruction.
///
/// \param __src
///    ZMM with 16 scale factors to write to BSR high half.
static __inline__ void __DEFAULT_FN_ATTRS_ACE _bsr0_movh_set(__m512i __src) {
  __builtin_ia32_bsrmovh_set((__v16si)__src);
}

/// Read the high half of BSR (scale factors 16-31) to a ZMM register.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> BSRMOVH </c> instruction.
///
/// \returns
///    ZMM containing 16 scale factors from BSR high half.
static __inline__ __m512i __DEFAULT_FN_ATTRS_ACE _bsr0_movh_get(void) {
  return (__m512i)__builtin_ia32_bsrmovh_get();
}

/// Load the low half of BSR (scale factors 0-15) from a ZMM register.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> BSRMOVL </c> instruction.
///
/// \param __src
///    ZMM with 16 scale factors to write to BSR low half.
static __inline__ void __DEFAULT_FN_ATTRS_ACE _bsr0_movl_set(__m512i __src) {
  __builtin_ia32_bsrmovl_set((__v16si)__src);
}

/// Read the low half of BSR (scale factors 0-15) to a ZMM register.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> BSRMOVL </c> instruction.
///
/// \returns
///    ZMM containing 16 scale factors from BSR low half.
static __inline__ __m512i __DEFAULT_FN_ATTRS_ACE _bsr0_movl_get(void) {
  return (__m512i)__builtin_ia32_bsrmovl_get();
}

/// Compute 2-way outer product of BF16 pairs, accumulating to FP32.
/// Each BF16 pair from src1 and src2 produces two FP32 products that
/// are accumulated into the destination tile.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TOP2BF16PS </c> instruction.
///
/// \param dst
///    Destination/accumulator tile register ID (0-7).
/// \param src1
///    First source ZMM register ID (0-31) containing 32 BF16 values.
/// \param src2
///    Second source ZMM register ID (0-31) containing 32 BF16 values.
#define _tile_top2bf16ps(dst, src1, src2)                                      \
  __builtin_ia32_top2bf16ps((dst), (src1), (src2))

/// Compute 4-way outer product of unsigned×unsigned bytes to INT32.
/// Each group of 4 unsigned byte pairs from src1 and src2 produces
/// 4 products accumulated into the destination tile as 32-bit integers.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TOP4BUUD </c> instruction.
///
/// \param dst
///    Destination/accumulator tile register ID (0-7).
/// \param src1
///    First source ZMM register ID (0-31) containing 64 unsigned bytes.
/// \param src2
///    Second source ZMM register ID (0-31) containing 64 unsigned bytes.
#define _tile_top4buud(dst, src1, src2)                                        \
  __builtin_ia32_top4buud((dst), (src1), (src2))

/// Compute 4-way outer product of unsigned×signed bytes to INT32.
/// Each group of 4 byte pairs (unsigned from src1, signed from src2)
/// produces 4 products accumulated into the destination tile.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TOP4BUSD </c> instruction.
///
/// \param dst
///    Destination/accumulator tile register ID (0-7).
/// \param src1
///    First source ZMM register ID (0-31) containing 64 unsigned bytes.
/// \param src2
///    Second source ZMM register ID (0-31) containing 64 signed bytes.
#define _tile_top4busd(dst, src1, src2)                                        \
  __builtin_ia32_top4busd((dst), (src1), (src2))

/// Compute 4-way outer product of signed×signed bytes to INT32.
/// Each group of 4 signed byte pairs produces 4 products accumulated
/// into the destination tile as 32-bit integers.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TOP4BSSD </c> instruction.
///
/// \param dst
///    Destination/accumulator tile register ID (0-7).
/// \param src1
///    First source ZMM register ID (0-31) containing 64 signed bytes.
/// \param src2
///    Second source ZMM register ID (0-31) containing 64 signed bytes.
#define _tile_top4bssd(dst, src1, src2)                                        \
  __builtin_ia32_top4bssd((dst), (src1), (src2))

/// Compute 4-way outer product of signed×unsigned bytes to INT32.
/// Each group of 4 byte pairs (signed from src1, unsigned from src2)
/// produces 4 products accumulated into the destination tile.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TOP4BSUD </c> instruction.
///
/// \param dst
///    Destination/accumulator tile register ID (0-7).
/// \param src1
///    First source ZMM register ID (0-31) containing 64 signed bytes.
/// \param src2
///    Second source ZMM register ID (0-31) containing 64 unsigned bytes.
#define _tile_top4bsud(dst, src1, src2)                                        \
  __builtin_ia32_top4bsud((dst), (src1), (src2))

/// Compute 4-way mixed precision outer product with HF8 (E4M3) format.
/// Multiplies HF8 values from src1 with BF8 values from src2, converting
/// to FP32 and accumulating into the destination tile.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TOP4MXHF8PS </c> instruction.
///
/// \param dst
///    Destination/accumulator tile register ID (0-7).
/// \param src1
///    First source ZMM register ID (0-31) containing HF8 values.
/// \param src2
///    Second source ZMM register ID (0-31) containing BF8 values.
/// \param imm
///    8-bit control immediate for scaling/rounding options.
#define _tile_top4mxhf8ps(dst, src1, src2, imm)                                \
  __builtin_ia32_top4mxhf8ps((dst), (src1), (src2), (imm))

/// Compute 4-way mixed precision outer product with BF8/HF8 format.
/// Multiplies BF8 values from src1 with HF8 (E4M3) values from src2,
/// converting to FP32 and accumulating into the destination tile.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TOP4MXBHF8PS </c> instruction.
///
/// \param dst
///    Destination/accumulator tile register ID (0-7).
/// \param src1
///    First source ZMM register ID (0-31) containing BF8 values.
/// \param src2
///    Second source ZMM register ID (0-31) containing HF8 values.
/// \param imm
///    8-bit control immediate for scaling/rounding options.
#define _tile_top4mxbhf8ps(dst, src1, src2, imm)                               \
  __builtin_ia32_top4mxbhf8ps((dst), (src1), (src2), (imm))

/// Compute 4-way mixed precision outer product with HF8/BF8 format.
/// Multiplies HF8 (E4M3) values from src1 with BF8 values from src2,
/// converting to FP32 and accumulating into the destination tile.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TOP4MXHBF8PS </c> instruction.
///
/// \param dst
///    Destination/accumulator tile register ID (0-7).
/// \param src1
///    First source ZMM register ID (0-31) containing HF8 values.
/// \param src2
///    Second source ZMM register ID (0-31) containing BF8 values.
/// \param imm
///    8-bit control immediate for scaling/rounding options.
#define _tile_top4mxhbf8ps(dst, src1, src2, imm)                               \
  __builtin_ia32_top4mxhbf8ps((dst), (src1), (src2), (imm))

/// Compute 4-way mixed precision outer product with BF8 (E5M2) format.
/// Multiplies BF8 values from both sources, converting to FP32 and
/// accumulating into the destination tile.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TOP4MXBF8PS </c> instruction.
///
/// \param dst
///    Destination/accumulator tile register ID (0-7).
/// \param src1
///    First source ZMM register ID (0-31) containing BF8 values.
/// \param src2
///    Second source ZMM register ID (0-31) containing BF8 values.
/// \param imm
///    8-bit control immediate for scaling/rounding options.
#define _tile_top4mxbf8ps(dst, src1, src2, imm)                                \
  __builtin_ia32_top4mxbf8ps((dst), (src1), (src2), (imm))

/// Compute 4-way mixed precision outer product of signed INT8 with BSR
/// scaling. Multiplies signed bytes, applies scale factors from the BSR,
/// and accumulates FP32 results into the destination tile.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TOP4MXBSSPS </c> instruction.
///
/// \param dst
///    Destination/accumulator tile register ID (0-7).
/// \param src1
///    First source ZMM register ID (0-31) containing signed bytes.
/// \param src2
///    Second source ZMM register ID (0-31) containing signed bytes.
/// \param imm
///    8-bit immediate selecting BSR scale factors to apply.
#define _tile_top4mxbssps(dst, src1, src2, imm)                                \
  __builtin_ia32_top4mxbssps((dst), (src1), (src2), (imm))

/// ACE tile type with fixed dimensions (16 rows × 64 bytes = 1024 bytes).
/// ACE Palette 2 uses fixed tile dimensions, unlike AMX Palette 1.
typedef int __acetile __attribute__((__vector_size__(1024), __aligned__(64)));

/// ACE Palette 2 tile configuration structure (64 bytes).
/// For ACE, only byte 0 (palette_id = 2) is significant; bytes 1-63
/// must be zero. Unlike AMX Palette 1, tile dimensions are fixed.
typedef struct __attribute__((__packed__, __aligned__(64))) {
  unsigned char palette_id;
  unsigned char reserved[63];
} __ace_tile_config;

/// Initialize an ACE tile configuration structure for Palette 2.
/// Sets palette_id to 2 and clears all reserved bytes to zero.
/// Use with _tile_ace_loadconfig to configure tiles for ACE operations.
static __inline__ void __DEFAULT_FN_ATTRS_ACE
__ace_init_config(__ace_tile_config *cfg) {
  for (int i = 0; i < 64; i++)
    ((unsigned char *)cfg)[i] = 0;
  cfg->palette_id = 2;
}

/// Zero an ACE tile variable. Sets all 1024 bytes to zero.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TILEZERO </c> instruction.
///
/// \param dst
///    Pointer to __acetile variable to be zeroed.
__DEFAULT_FN_ATTRS_ACE
static __inline__ void __tile_ace_zero(__acetile *dst) {
  *dst = __builtin_ia32_tilezero_internal(16, 64);
}

/// Compute 4-way outer product of unsigned×unsigned bytes to INT32.
/// Multiplies 64 unsigned bytes from each ZMM source, producing a
/// 16×16 grid of 32-bit sums accumulated into the ACE tile.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TOP4BUUD </c> instruction.
///
/// \param dst
///    Pointer to destination/accumulator __acetile.
/// \param src1
///    First source ZMM vector containing 64 unsigned bytes.
/// \param src2
///    Second source ZMM vector containing 64 unsigned bytes.
__DEFAULT_FN_ATTRS_ACE
static __inline__ void __tile_ace_top4buud(__acetile *dst, __m512i src1,
                                           __m512i src2) {
  *dst = __builtin_ia32_top4buud_internal(16, 64, 64, *dst, (__v64qs)src1,
                                          (__v64qs)src2);
}

/// Compute 4-way outer product of unsigned×signed bytes to INT32.
/// Multiplies 64 bytes from each ZMM source (unsigned from src1,
/// signed from src2), producing a 16×16 grid of 32-bit sums.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TOP4BUSD </c> instruction.
///
/// \param dst
///    Pointer to destination/accumulator __acetile.
/// \param src1
///    First source ZMM vector containing 64 unsigned bytes.
/// \param src2
///    Second source ZMM vector containing 64 signed bytes.
__DEFAULT_FN_ATTRS_ACE
static __inline__ void __tile_ace_top4busd(__acetile *dst, __m512i src1,
                                           __m512i src2) {
  *dst = __builtin_ia32_top4busd_internal(16, 64, 64, *dst, (__v64qs)src1,
                                          (__v64qs)src2);
}

/// Compute 4-way outer product of signed×signed bytes to INT32.
/// Multiplies 64 signed bytes from each ZMM source, producing a
/// 16×16 grid of 32-bit sums accumulated into the ACE tile.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TOP4BSSD </c> instruction.
///
/// \param dst
///    Pointer to destination/accumulator __acetile.
/// \param src1
///    First source ZMM vector containing 64 signed bytes.
/// \param src2
///    Second source ZMM vector containing 64 signed bytes.
__DEFAULT_FN_ATTRS_ACE
static __inline__ void __tile_ace_top4bssd(__acetile *dst, __m512i src1,
                                           __m512i src2) {
  *dst = __builtin_ia32_top4bssd_internal(16, 64, 64, *dst, (__v64qs)src1,
                                          (__v64qs)src2);
}

/// Compute 4-way outer product of signed×unsigned bytes to INT32.
/// Multiplies 64 bytes from each ZMM source (signed from src1,
/// unsigned from src2), producing a 16×16 grid of 32-bit sums.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TOP4BSUD </c> instruction.
///
/// \param dst
///    Pointer to destination/accumulator __acetile.
/// \param src1
///    First source ZMM vector containing 64 signed bytes.
/// \param src2
///    Second source ZMM vector containing 64 unsigned bytes.
__DEFAULT_FN_ATTRS_ACE
static __inline__ void __tile_ace_top4bsud(__acetile *dst, __m512i src1,
                                           __m512i src2) {
  *dst = __builtin_ia32_top4bsud_internal(16, 64, 64, *dst, (__v64qs)src1,
                                          (__v64qs)src2);
}

/// Compute 2-way outer product of BF16 to FP32.
/// Multiplies 32 BF16 values from each source, producing a 16×16 grid
/// of FP32 sums accumulated into the ACE tile.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TOP2BF16PS </c> instruction.
///
/// \param dst
///    Pointer to destination/accumulator __acetile.
/// \param src1
///    First source ZMM vector (__m512bh) containing 32 BF16 values.
/// \param src2
///    Second source ZMM vector (__m512bh) containing 32 BF16 values.
__DEFAULT_FN_ATTRS_ACE
static __inline__ void __tile_ace_top2bf16ps(__acetile *dst, __m512bh src1,
                                             __m512bh src2) {
  *dst = __builtin_ia32_top2bf16ps_internal(16, 64, 64, *dst, (__v32bf)src1,
                                            (__v32bf)src2);
}

/// Compute 4-way mixed precision outer product with HF8 (E4M3) format.
/// Multiplies HF8 values from src1 with HF8 values from src2, applies
/// BSR scaling, converts to FP32 and accumulates into the ACE tile.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TOP4MXHF8PS </c> instruction.
///
/// \param dst
///    Pointer to destination/accumulator __acetile.
/// \param src1
///    First source ZMM vector containing HF8 (E4M3) values.
/// \param src2
///    Second source ZMM vector containing HF8 (E4M3) values.
/// \param imm
///    8-bit immediate selecting BSR scale factors to apply.
#define __tile_ace_top4mxhf8ps(dst, src1, src2, imm)                           \
  (*(dst) = __builtin_ia32_top4mxhf8ps_nobsr_internal(                         \
       16, 64, 64, (imm), *(dst), (__v16si)(src1), (__v16si)(src2)))

/// Compute 4-way mixed precision outer product with BF8/HF8 format.
/// Multiplies BF8 (E5M2) values from src1 with HF8 (E4M3) values from src2,
/// applies BSR scaling, converts to FP32 and accumulates into the ACE tile.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TOP4MXBHF8PS </c> instruction.
///
/// \param dst
///    Pointer to destination/accumulator __acetile.
/// \param src1
///    First source ZMM vector containing BF8 (E5M2) values.
/// \param src2
///    Second source ZMM vector containing HF8 (E4M3) values.
/// \param imm
///    8-bit immediate selecting BSR scale factors to apply.
#define __tile_ace_top4mxbhf8ps(dst, src1, src2, imm)                          \
  (*(dst) = __builtin_ia32_top4mxbhf8ps_nobsr_internal(                        \
       16, 64, 64, (imm), *(dst), (__v16si)(src1), (__v16si)(src2)))

/// Compute 4-way mixed precision outer product with HF8/BF8 format.
/// Multiplies HF8 (E4M3) values from src1 with BF8 (E5M2) values from src2,
/// applies BSR scaling, converts to FP32 and accumulates into the ACE tile.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TOP4MXHBF8PS </c> instruction.
///
/// \param dst
///    Pointer to destination/accumulator __acetile.
/// \param src1
///    First source ZMM vector containing HF8 (E4M3) values.
/// \param src2
///    Second source ZMM vector containing BF8 (E5M2) values.
/// \param imm
///    8-bit immediate selecting BSR scale factors to apply.
#define __tile_ace_top4mxhbf8ps(dst, src1, src2, imm)                          \
  (*(dst) = __builtin_ia32_top4mxhbf8ps_nobsr_internal(                        \
       16, 64, 64, (imm), *(dst), (__v16si)(src1), (__v16si)(src2)))

/// Compute 4-way mixed precision outer product with BF8 (E5M2) format.
/// Multiplies BF8 values from both sources, applies BSR scaling,
/// converts to FP32 and accumulates into the ACE tile.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TOP4MXBF8PS </c> instruction.
///
/// \param dst
///    Pointer to destination/accumulator __acetile.
/// \param src1
///    First source ZMM vector containing BF8 (E5M2) values.
/// \param src2
///    Second source ZMM vector containing BF8 (E5M2) values.
/// \param imm
///    8-bit immediate selecting BSR scale factors to apply.
#define __tile_ace_top4mxbf8ps(dst, src1, src2, imm)                           \
  (*(dst) = __builtin_ia32_top4mxbf8ps_nobsr_internal(                         \
       16, 64, 64, (imm), *(dst), (__v16si)(src1), (__v16si)(src2)))

/// Compute 4-way mixed precision outer product of MX INT8 with BSR scaling.
/// Multiplies signed MX INT8 bytes from both sources, applies BSR scaling,
/// converts to FP32 and accumulates into the ACE tile.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TOP4MXBSSPS </c> instruction.
///
/// \param dst
///    Pointer to destination/accumulator __acetile.
/// \param src1
///    First source ZMM vector containing signed MX INT8 values.
/// \param src2
///    Second source ZMM vector containing signed MX INT8 values.
/// \param imm
///    8-bit immediate selecting BSR scale factors to apply.
#define __tile_ace_top4mxbssps(dst, src1, src2, imm)                           \
  (*(dst) = __builtin_ia32_top4mxbssps_nobsr_internal(                         \
       16, 64, 64, (imm), *(dst), (__v16si)(src1), (__v16si)(src2)))

/// Write a ZMM vector as a column in an ACE tile.
/// The 16 doublewords from src are written vertically at column idx.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TILEMOVCOL </c> instruction.
///
/// \param dst
///    Pointer to destination __acetile.
/// \param src
///    Source ZMM vector (__m512i) with 16 doublewords.
/// \param idx
///    Column index (0-15).
__DEFAULT_FN_ATTRS_ACE
static __inline__ void __tile_ace_setcol(__acetile *dst, __m512i src,
                                         unsigned int idx) {
  *dst = __builtin_ia32_tilesetcol_internal(16, 64, (__v16si)src, idx);
}

/// Write a ZMM vector as a row in an ACE tile.
/// The 64 bytes from src are written horizontally at row idx.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TILEMOVROW </c> instruction.
///
/// \param dst
///    Pointer to destination __acetile.
/// \param src
///    Source ZMM vector (__m512i) with 64 bytes.
/// \param idx
///    Row index (0-15).
__DEFAULT_FN_ATTRS_ACE
static __inline__ void __tile_ace_setrow(__acetile *dst, __m512i src,
                                         unsigned int idx) {
  *dst = __builtin_ia32_tilesetrow_internal(16, 64, (__v16si)src, idx);
}

/// Read a row from an ACE tile to a ZMM vector.
/// Returns the 64 bytes at row idx as a ZMM vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TILEMOVROW </c> instruction.
///
/// \param src
///    Pointer to source __acetile.
/// \param idx
///    Row index (0-15).
/// \returns
///    ZMM vector (__m512i) containing the 64-byte row.
__DEFAULT_FN_ATTRS_ACE
static __inline__ __m512i __tile_ace_getrow(__acetile *src, unsigned int idx) {
  return (__m512i)__builtin_ia32_tilemovrow_internal(16, 64, *src, idx);
}

/// Read a row from an ACE tile and convert INT32 elements to FP32.
/// Each of the 16 INT32 elements in the row is converted to FP32.
/// Rounding uses RTNE (round to nearest even).
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TCVTROWD2PS </c> instruction.
///
/// \param src
///    Pointer to source __acetile containing INT32 elements.
/// \param idx
///    Row index (0-15).
/// \returns
///    ZMM vector (__m512) containing 16 FP32 values.
__DEFAULT_FN_ATTRS_ACE
static __inline__ __m512 __tile_ace_cvtrowd2ps(__acetile *src,
                                               unsigned int idx) {
  return __builtin_ia32_tcvtrowd2ps_internal(16, 64, *src, idx);
}

/// Read a row from an ACE tile and convert FP32 elements to BF16 (high).
/// Each FP32 element is converted to BF16 and placed in the high 16 bits
/// of each destination dword; low 16 bits are zeroed.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TCVTROWPS2BF16H </c> instruction.
///
/// \param src
///    Pointer to source __acetile containing FP32 elements.
/// \param idx
///    Row index (0-15).
/// \returns
///    ZMM vector (__m512bh) with BF16 values in high half of each dword.
__DEFAULT_FN_ATTRS_ACE
static __inline__ __m512bh __tile_ace_cvtrowps2bf16h(__acetile *src,
                                                     unsigned int idx) {
  return __builtin_ia32_tcvtrowps2bf16h_internal(16, 64, *src, idx);
}

/// Read a row from an ACE tile and convert FP32 elements to BF16 (low).
/// Each FP32 element is converted to BF16 and placed in the low 16 bits
/// of each destination dword; high 16 bits are zeroed.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TCVTROWPS2BF16L </c> instruction.
///
/// \param src
///    Pointer to source __acetile containing FP32 elements.
/// \param idx
///    Row index (0-15).
/// \returns
///    ZMM vector (__m512bh) with BF16 values in low half of each dword.
__DEFAULT_FN_ATTRS_ACE
static __inline__ __m512bh __tile_ace_cvtrowps2bf16l(__acetile *src,
                                                     unsigned int idx) {
  return __builtin_ia32_tcvtrowps2bf16l_internal(16, 64, *src, idx);
}

/// Read a row from an ACE tile and convert FP32 elements to FP16 (high).
/// Each FP32 element is converted to FP16 and placed in the high 16 bits
/// of each destination dword; low 16 bits are zeroed.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TCVTROWPS2PHH </c> instruction.
///
/// \param src
///    Pointer to source __acetile containing FP32 elements.
/// \param idx
///    Row index (0-15).
/// \returns
///    ZMM vector (__m512h) with FP16 values in high half of each dword.
__DEFAULT_FN_ATTRS_ACE
static __inline__ __m512h __tile_ace_cvtrowps2phh(__acetile *src,
                                                  unsigned int idx) {
  return __builtin_ia32_tcvtrowps2phh_internal(16, 64, *src, idx);
}

/// Read a row from an ACE tile and convert FP32 elements to FP16 (low).
/// Each FP32 element is converted to FP16 and placed in the low 16 bits
/// of each destination dword; high 16 bits are zeroed.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TCVTROWPS2PHL </c> instruction.
///
/// \param src
///    Pointer to source __acetile containing FP32 elements.
/// \param idx
///    Row index (0-15).
/// \returns
///    ZMM vector (__m512h) with FP16 values in low half of each dword.
__DEFAULT_FN_ATTRS_ACE
static __inline__ __m512h __tile_ace_cvtrowps2phl(__acetile *src,
                                                  unsigned int idx) {
  return __builtin_ia32_tcvtrowps2phl_internal(16, 64, *src, idx);
}

/// Compute 4-way mixed precision outer product with HF8 (E4M3) format
/// using explicit BSR scales. Multiplies HF8 values from src1 with HF8
/// values from src2, applies scales from the __bsr struct, converts to
/// FP32 and accumulates into the ACE tile.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TOP4MXHF8PS </c> instruction.
///
/// \param dst
///    Pointer to destination/accumulator __acetile.
/// \param src1
///    First source ZMM vector containing HF8 (E4M3) values.
/// \param src2
///    Second source ZMM vector containing HF8 (E4M3) values.
/// \param scales
///    __bsr struct containing A-scales (hi) and B-scales (lo).
/// \param imm
///    8-bit immediate selecting BSR scale factors to apply.
#define __tile_ace_top4mxhf8ps_bsr(dst, src1, src2, scales, imm)               \
  (*(dst) = __builtin_ia32_top4mxhf8ps_internal(                               \
       16, 64, 64, (imm), *(dst), (__v16si)(src1), (__v16si)(src2),            \
       __bsr_combine_v32((__v16si)((scales).lo), (__v16si)((scales).hi))))

/// Compute 4-way mixed precision outer product with BF8/HF8 format
/// using explicit BSR scales. Multiplies BF8 (E5M2) values from src1
/// with HF8 (E4M3) values from src2, applies scales from the __bsr struct,
/// converts to FP32 and accumulates into the ACE tile.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TOP4MXBHF8PS </c> instruction.
///
/// \param dst
///    Pointer to destination/accumulator __acetile.
/// \param src1
///    First source ZMM vector containing BF8 (E5M2) values.
/// \param src2
///    Second source ZMM vector containing HF8 (E4M3) values.
/// \param scales
///    __bsr struct containing A-scales (hi) and B-scales (lo).
/// \param imm
///    8-bit immediate selecting BSR scale factors to apply.
#define __tile_ace_top4mxbhf8ps_bsr(dst, src1, src2, scales, imm)              \
  (*(dst) = __builtin_ia32_top4mxbhf8ps_internal(                              \
       16, 64, 64, (imm), *(dst), (__v16si)(src1), (__v16si)(src2),            \
       __bsr_combine_v32((__v16si)((scales).lo), (__v16si)((scales).hi))))

/// Compute 4-way mixed precision outer product with HF8/BF8 format
/// using explicit BSR scales. Multiplies HF8 (E4M3) values from src1
/// with BF8 (E5M2) values from src2, applies scales from the __bsr struct,
/// converts to FP32 and accumulates into the ACE tile.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TOP4MXHBF8PS </c> instruction.
///
/// \param dst
///    Pointer to destination/accumulator __acetile.
/// \param src1
///    First source ZMM vector containing HF8 (E4M3) values.
/// \param src2
///    Second source ZMM vector containing BF8 (E5M2) values.
/// \param scales
///    __bsr struct containing A-scales (hi) and B-scales (lo).
/// \param imm
///    8-bit immediate selecting BSR scale factors to apply.
#define __tile_ace_top4mxhbf8ps_bsr(dst, src1, src2, scales, imm)              \
  (*(dst) = __builtin_ia32_top4mxhbf8ps_internal(                              \
       16, 64, 64, (imm), *(dst), (__v16si)(src1), (__v16si)(src2),            \
       __bsr_combine_v32((__v16si)((scales).lo), (__v16si)((scales).hi))))

/// Compute 4-way mixed precision outer product with BF8 (E5M2) format
/// using explicit BSR scales. Multiplies BF8 values from both sources,
/// applies scales from the __bsr struct, converts to FP32 and accumulates
/// into the ACE tile.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TOP4MXBF8PS </c> instruction.
///
/// \param dst
///    Pointer to destination/accumulator __acetile.
/// \param src1
///    First source ZMM vector containing BF8 (E5M2) values.
/// \param src2
///    Second source ZMM vector containing BF8 (E5M2) values.
/// \param scales
///    __bsr struct containing A-scales (hi) and B-scales (lo).
/// \param imm
///    8-bit immediate selecting BSR scale factors to apply.
#define __tile_ace_top4mxbf8ps_bsr(dst, src1, src2, scales, imm)               \
  (*(dst) = __builtin_ia32_top4mxbf8ps_internal(                               \
       16, 64, 64, (imm), *(dst), (__v16si)(src1), (__v16si)(src2),            \
       __bsr_combine_v32((__v16si)((scales).lo), (__v16si)((scales).hi))))

#undef __DEFAULT_FN_ATTRS_ACE

#endif /* __x86_64__ */
#endif /* __ACEV1INTRIN_H */
