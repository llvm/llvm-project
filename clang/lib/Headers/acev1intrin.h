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
#if defined(__x86_64__) && defined(__SSE2__)

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS_ACE                                                 \
  __attribute__((__always_inline__, __nodebug__, __target__("acev1")))

// clang-format off

/// Load tile configuration from a 64-byte memory location. For ACE
/// (Palette 2), the palette_id byte must be 2. Unlike AMX (Palette 1),
/// ACE tiles have fixed dimensions of 16 rows x 64 bytes, so per-tile
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
/// configuration bytes reflect the fixed 16x64 dimensions. If tiles
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
static __inline__ void __DEFAULT_FN_ATTRS_ACE
_tile_ace_release(void) {
  __builtin_ia32_tilerelease();
}

/// Zero the ACE tile specified by "tile". Sets all 1024 bytes
/// (16 rows x 64 bytes) of the tile register to zero.
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
///    Source ZMM vector (__m512i) containing 16 doublewords.
/// \param idx
///    Column index (0-15). Immediate or register form selected automatically.
#define _tile_insertcol(dst, src, idx)                                         \
  __builtin_ia32_tilemovcolinsert((dst), (__v16si)(src), (idx))

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
///    Source ZMM vector (__m512i) containing 16 doublewords.
/// \param idx
///    Row index (0-15). Immediate or register form selected automatically.
#define _tile_insertrow(dst, src, idx)                                         \
  __builtin_ia32_tilemovrowinsert((dst), (__v16si)(src), (idx))

/// Initialize the Block Scale Register (BSR), setting all 128 scale bytes to
/// 0x7F (the E8M0 encoding of 1.0), the same state LDTILECFG leaves it in.
/// The BSR holds the scale factors used by the mixed-precision outer product
/// instructions (TOP4MX* variants).
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> BSRINIT </c> instruction.
static __inline__ void __DEFAULT_FN_ATTRS_ACE
_bsr0_init(void) {
  __builtin_ia32_bsr0init();
}

/// Load the full BSR (128 scale bytes) from two ZMM registers. The high half
/// (A scales) comes from __src1 and the low half (B scales) from __src2.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> BSRMOVF </c> instruction.
///
/// \param __src1
///    ZMM with the 64 A-operand scale bytes for the BSR high half.
/// \param __src2
///    ZMM with the 64 B-operand scale bytes for the BSR low half.
static __inline__ void __DEFAULT_FN_ATTRS_ACE
_bsr0_insertfull(__m512i __src1, __m512i __src2) {
  __builtin_ia32_bsr0movf((__v64qi)__src1, (__v64qi)__src2);
}

/// Load the high half of BSR (64 A-operand scale bytes) from a ZMM register.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> BSRMOVH </c> instruction.
///
/// \param __src
///    ZMM with 64 scale bytes to write to the BSR high half.
static __inline__ void __DEFAULT_FN_ATTRS_ACE
_bsr0_inserth(__m512i __src) {
  __builtin_ia32_bsr0movhinsert((__v64qi)__src);
}

/// Read the high half of BSR (64 A-operand scale bytes) to a ZMM register.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> BSRMOVH </c> instruction.
///
/// \returns
///    ZMM containing the 64 scale bytes from the BSR high half.
static __inline__ __m512i __DEFAULT_FN_ATTRS_ACE
_bsr0_extracth(void) {
  return (__m512i)__builtin_ia32_bsr0movhextract();
}

/// Load the low half of BSR (64 B-operand scale bytes) from a ZMM register.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> BSRMOVL </c> instruction.
///
/// \param __src
///    ZMM with 64 scale bytes to write to the BSR low half.
static __inline__ void __DEFAULT_FN_ATTRS_ACE
_bsr0_insertl(__m512i __src) {
  __builtin_ia32_bsr0movlinsert((__v64qi)__src);
}

/// Read the low half of BSR (64 B-operand scale bytes) to a ZMM register.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> BSRMOVL </c> instruction.
///
/// \returns
///    ZMM containing the 64 scale bytes from the BSR low half.
static __inline__ __m512i __DEFAULT_FN_ATTRS_ACE
_bsr0_extractl(void) {
  return (__m512i)__builtin_ia32_bsr0movlextract();
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
///    First source ZMM vector (__m512bh) containing 32 BF16 values.
/// \param src2
///    Second source ZMM vector (__m512bh) containing 32 BF16 values.
#define _tile_op2bf16_ps(dst, src1, src2)                                      \
  __builtin_ia32_top2bf16ps((dst), (__v32bf)(src1), (__v32bf)(src2))

/// Compute 4-way outer product of unsigned x unsigned bytes to INT32.
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
///    First source ZMM vector (__m512i) containing 64 unsigned bytes.
/// \param src2
///    Second source ZMM vector (__m512i) containing 64 unsigned bytes.
#define _tile_op4buud_epi32(dst, src1, src2)                                   \
  __builtin_ia32_top4buud((dst), (__v64qi)(src1), (__v64qi)(src2))

/// Compute 4-way outer product of unsigned x signed bytes to INT32.
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
///    First source ZMM vector (__m512i) containing 64 unsigned bytes.
/// \param src2
///    Second source ZMM vector (__m512i) containing 64 signed bytes.
#define _tile_op4busd_epi32(dst, src1, src2)                                   \
  __builtin_ia32_top4busd((dst), (__v64qi)(src1), (__v64qi)(src2))

/// Compute 4-way outer product of signed x signed bytes to INT32.
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
///    First source ZMM vector (__m512i) containing 64 signed bytes.
/// \param src2
///    Second source ZMM vector (__m512i) containing 64 signed bytes.
#define _tile_op4bssd_epi32(dst, src1, src2)                                   \
  __builtin_ia32_top4bssd((dst), (__v64qi)(src1), (__v64qi)(src2))

/// Compute 4-way outer product of signed x unsigned bytes to INT32.
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
///    First source ZMM vector (__m512i) containing 64 signed bytes.
/// \param src2
///    Second source ZMM vector (__m512i) containing 64 unsigned bytes.
#define _tile_op4bsud_epi32(dst, src1, src2)                                   \
  __builtin_ia32_top4bsud((dst), (__v64qi)(src1), (__v64qi)(src2))

/// Select the A-input scale group within the BSR for the mixed precision
/// outer product intrinsics. Combine with \c _MM_ACE_SCALE_B using a bitwise
/// OR to form the scale group selector.
///
/// \headerfile <immintrin.h>
///
/// \param g
///    A-input scale group number (0-3).
/// \returns
///    The A-input field of the scale group selector.
#define _MM_ACE_SCALE_A(g) ((g) << 0)

/// Select the B-input scale group within the BSR for the mixed precision
/// outer product intrinsics. Combine with \c _MM_ACE_SCALE_A using a bitwise
/// OR to form the scale group selector.
///
/// \headerfile <immintrin.h>
///
/// \param g
///    B-input scale group number (0-3).
/// \returns
///    The B-input field of the scale group selector.
#define _MM_ACE_SCALE_B(g) ((g) << 3)

/// Rank-4 MX FP8 outer product. A = FP8 E4M3, B = FP8 E4M3. OCP MX block
/// scaling via the BSR, with FP32 accumulation into the destination tile.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TOP4MXHF8PS </c> instruction.
///
/// \param dst
///    Destination/accumulator tile register ID (0-7).
/// \param src1
///    The A input: ZMM vector (__m512i) of FP8 E4M3 (HF8) values.
/// \param src2
///    The B input: ZMM vector (__m512i) of FP8 E4M3 (HF8) values.
/// \param imm
///    Scale group selector, formed from \c _MM_ACE_SCALE_A and
///    \c _MM_ACE_SCALE_B; all other bits are reserved and must be zero.
#define _tile_op4mxhf8_ps(dst, src1, src2, imm)                                \
  __builtin_ia32_top4mxhf8ps((dst), (__v64qi)(src1), (__v64qi)(src2), (imm))

/// Rank-4 MX FP8 outer product. A = FP8 E5M2, B = FP8 E4M3. OCP MX block
/// scaling via the BSR, with FP32 accumulation into the destination tile.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TOP4MXBHF8PS </c> instruction.
///
/// \param dst
///    Destination/accumulator tile register ID (0-7).
/// \param src1
///    The A input: ZMM vector (__m512i) of FP8 E5M2 (BF8) values.
/// \param src2
///    The B input: ZMM vector (__m512i) of FP8 E4M3 (HF8) values.
/// \param imm
///    Scale group selector, formed from \c _MM_ACE_SCALE_A and
///    \c _MM_ACE_SCALE_B; all other bits are reserved and must be zero.
#define _tile_op4mxbhf8_ps(dst, src1, src2, imm)                               \
  __builtin_ia32_top4mxbhf8ps((dst), (__v64qi)(src1), (__v64qi)(src2), (imm))

/// Rank-4 MX FP8 outer product. A = FP8 E4M3, B = FP8 E5M2. OCP MX block
/// scaling via the BSR, with FP32 accumulation into the destination tile.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TOP4MXHBF8PS </c> instruction.
///
/// \param dst
///    Destination/accumulator tile register ID (0-7).
/// \param src1
///    The A input: ZMM vector (__m512i) of FP8 E4M3 (HF8) values.
/// \param src2
///    The B input: ZMM vector (__m512i) of FP8 E5M2 (BF8) values.
/// \param imm
///    Scale group selector, formed from \c _MM_ACE_SCALE_A and
///    \c _MM_ACE_SCALE_B; all other bits are reserved and must be zero.
#define _tile_op4mxhbf8_ps(dst, src1, src2, imm)                               \
  __builtin_ia32_top4mxhbf8ps((dst), (__v64qi)(src1), (__v64qi)(src2), (imm))

/// Rank-4 MX FP8 outer product. A = FP8 E5M2, B = FP8 E5M2. OCP MX block
/// scaling via the BSR, with FP32 accumulation into the destination tile.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TOP4MXBF8PS </c> instruction.
///
/// \param dst
///    Destination/accumulator tile register ID (0-7).
/// \param src1
///    The A input: ZMM vector (__m512i) of FP8 E5M2 (BF8) values.
/// \param src2
///    The B input: ZMM vector (__m512i) of FP8 E5M2 (BF8) values.
/// \param imm
///    Scale group selector, formed from \c _MM_ACE_SCALE_A and
///    \c _MM_ACE_SCALE_B; all other bits are reserved and must be zero.
#define _tile_op4mxbf8_ps(dst, src1, src2, imm)                                \
  __builtin_ia32_top4mxbf8ps((dst), (__v64qi)(src1), (__v64qi)(src2), (imm))

/// Rank-4 MX INT8 outer product. A = MX INT8 signed, B = MX INT8 signed.
/// OCP MX block scaling via the BSR, with FP32 accumulation into the
/// destination tile.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TOP4MXBSSPS </c> instruction.
///
/// \param dst
///    Destination/accumulator tile register ID (0-7).
/// \param src1
///    The A input: ZMM vector (__m512i) of MX INT8 signed values.
/// \param src2
///    The B input: ZMM vector (__m512i) of MX INT8 signed values.
/// \param imm
///    Scale group selector, formed from \c _MM_ACE_SCALE_A and
///    \c _MM_ACE_SCALE_B; all other bits are reserved and must be zero.
#define _tile_op4mxbss_ps(dst, src1, src2, imm)                                \
  __builtin_ia32_top4mxbssps((dst), (__v64qi)(src1), (__v64qi)(src2), (imm))

/// Read a row from a tile register and convert its int32 elements to FP32.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TCVTROWD2PS </c> instruction.
///
/// \param tsrc
///    Source tile register ID (0-7).
/// \param row
///    Row index selecting the tile row to read.
/// \returns
///    ZMM vector (__m512) holding the converted FP32 elements.
#define _tile_cvtrow_epi32_ps(tsrc, row) _tile_cvtrowd2ps((tsrc), (row))

/// Read a row from a tile register and convert its FP32 elements to BF16,
/// placing the results in the high half of each destination dword.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TCVTROWPS2BF16H </c> instruction.
///
/// \param tsrc
///    Source tile register ID (0-7).
/// \param row
///    Row index selecting the tile row to read.
/// \returns
///    ZMM vector (__m512bh) holding the converted BF16 elements.
#define _tile_cvtrowh_ps_pbh(tsrc, row) _tile_cvtrowps2bf16h((tsrc), (row))

/// Read a row from a tile register and convert its FP32 elements to BF16,
/// placing the results in the low half of each destination dword.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TCVTROWPS2BF16L </c> instruction.
///
/// \param tsrc
///    Source tile register ID (0-7).
/// \param row
///    Row index selecting the tile row to read.
/// \returns
///    ZMM vector (__m512bh) holding the converted BF16 elements.
#define _tile_cvtrowl_ps_pbh(tsrc, row) _tile_cvtrowps2bf16l((tsrc), (row))

/// Read a row from a tile register and convert its FP32 elements to FP16,
/// placing the results in the high half of each destination dword.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TCVTROWPS2PHH </c> instruction.
///
/// \param tsrc
///    Source tile register ID (0-7).
/// \param row
///    Row index selecting the tile row to read.
/// \returns
///    ZMM vector (__m512h) holding the converted FP16 elements.
#define _tile_cvtrowh_ps_ph(tsrc, row) _tile_cvtrowps2phh((tsrc), (row))

/// Read a row from a tile register and convert its FP32 elements to FP16,
/// placing the results in the low half of each destination dword.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TCVTROWPS2PHL </c> instruction.
///
/// \param tsrc
///    Source tile register ID (0-7).
/// \param row
///    Row index selecting the tile row to read.
/// \returns
///    ZMM vector (__m512h) holding the converted FP16 elements.
#define _tile_cvtrowl_ps_ph(tsrc, row) _tile_cvtrowps2phl((tsrc), (row))

/// Extract one row of a tile register into a ZMM destination.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TILEMOVROW </c> instruction.
///
/// \param tsrc
///    Source tile register ID (0-7).
/// \param row
///    Row index selecting the tile row to read.
/// \returns
///    ZMM vector (__m512i) holding the extracted row.
#define _tile_extractrow(tsrc, row) _tile_movrow((tsrc), (row))

/// ACE tile type with fixed dimensions (16 rows x 64 bytes = 1024 bytes).
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
  __builtin_memset(cfg, 0, sizeof(*cfg));
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

/// Compute 4-way outer product of unsigned x unsigned bytes to INT32.
/// Multiplies 64 unsigned bytes from each ZMM source, producing a
/// 16x16 grid of 32-bit sums accumulated into the ACE tile.
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
  *dst = __builtin_ia32_top4buud_internal(16, 64, 64, *dst, (__v64qi)src1,
                                          (__v64qi)src2);
}

/// Compute 4-way outer product of unsigned x signed bytes to INT32.
/// Multiplies 64 bytes from each ZMM source (unsigned from src1,
/// signed from src2), producing a 16x16 grid of 32-bit sums.
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
  *dst = __builtin_ia32_top4busd_internal(16, 64, 64, *dst, (__v64qi)src1,
                                          (__v64qi)src2);
}

/// Compute 4-way outer product of signed x signed bytes to INT32.
/// Multiplies 64 signed bytes from each ZMM source, producing a
/// 16x16 grid of 32-bit sums accumulated into the ACE tile.
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
  *dst = __builtin_ia32_top4bssd_internal(16, 64, 64, *dst, (__v64qi)src1,
                                          (__v64qi)src2);
}

/// Compute 4-way outer product of signed x unsigned bytes to INT32.
/// Multiplies 64 bytes from each ZMM source (signed from src1,
/// unsigned from src2), producing a 16x16 grid of 32-bit sums.
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
  *dst = __builtin_ia32_top4bsud_internal(16, 64, 64, *dst, (__v64qi)src1,
                                          (__v64qi)src2);
}

/// Compute 2-way outer product of BF16 to FP32.
/// Multiplies 32 BF16 values from each source, producing a 16x16 grid
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

/// Rank-4 MX FP8 outer product. A = FP8 E4M3, B = FP8 E4M3. OCP MX block
/// scaling via the BSR, with FP32 accumulation into the destination tile.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TOP4MXHF8PS </c> instruction.
///
/// \param dst
///    Pointer to destination/accumulator __acetile.
/// \param src1
///    The A input: ZMM vector of FP8 E4M3 (HF8) values.
/// \param src2
///    The B input: ZMM vector of FP8 E4M3 (HF8) values.
/// \param imm
///    Scale group selector, formed from \c _MM_ACE_SCALE_A and
///    \c _MM_ACE_SCALE_B; all other bits are reserved and must be zero.
#define __tile_ace_top4mxhf8ps(dst, src1, src2, imm)                           \
  (*(dst) = __builtin_ia32_top4mxhf8ps_internal(                               \
       16, 64, 64, (imm), *(dst), (__v64qi)(src1), (__v64qi)(src2)))

/// Rank-4 MX FP8 outer product. A = FP8 E5M2, B = FP8 E4M3. OCP MX block
/// scaling via the BSR, with FP32 accumulation into the destination tile.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TOP4MXBHF8PS </c> instruction.
///
/// \param dst
///    Pointer to destination/accumulator __acetile.
/// \param src1
///    The A input: ZMM vector of FP8 E5M2 (BF8) values.
/// \param src2
///    The B input: ZMM vector of FP8 E4M3 (HF8) values.
/// \param imm
///    Scale group selector, formed from \c _MM_ACE_SCALE_A and
///    \c _MM_ACE_SCALE_B; all other bits are reserved and must be zero.
#define __tile_ace_top4mxbhf8ps(dst, src1, src2, imm)                          \
  (*(dst) = __builtin_ia32_top4mxbhf8ps_internal(                              \
       16, 64, 64, (imm), *(dst), (__v64qi)(src1), (__v64qi)(src2)))

/// Rank-4 MX FP8 outer product. A = FP8 E4M3, B = FP8 E5M2. OCP MX block
/// scaling via the BSR, with FP32 accumulation into the destination tile.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TOP4MXHBF8PS </c> instruction.
///
/// \param dst
///    Pointer to destination/accumulator __acetile.
/// \param src1
///    The A input: ZMM vector of FP8 E4M3 (HF8) values.
/// \param src2
///    The B input: ZMM vector of FP8 E5M2 (BF8) values.
/// \param imm
///    Scale group selector, formed from \c _MM_ACE_SCALE_A and
///    \c _MM_ACE_SCALE_B; all other bits are reserved and must be zero.
#define __tile_ace_top4mxhbf8ps(dst, src1, src2, imm)                          \
  (*(dst) = __builtin_ia32_top4mxhbf8ps_internal(                              \
       16, 64, 64, (imm), *(dst), (__v64qi)(src1), (__v64qi)(src2)))

/// Rank-4 MX FP8 outer product. A = FP8 E5M2, B = FP8 E5M2. OCP MX block
/// scaling via the BSR, with FP32 accumulation into the destination tile.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TOP4MXBF8PS </c> instruction.
///
/// \param dst
///    Pointer to destination/accumulator __acetile.
/// \param src1
///    The A input: ZMM vector of FP8 E5M2 (BF8) values.
/// \param src2
///    The B input: ZMM vector of FP8 E5M2 (BF8) values.
/// \param imm
///    Scale group selector, formed from \c _MM_ACE_SCALE_A and
///    \c _MM_ACE_SCALE_B; all other bits are reserved and must be zero.
#define __tile_ace_top4mxbf8ps(dst, src1, src2, imm)                           \
  (*(dst) = __builtin_ia32_top4mxbf8ps_internal(                               \
       16, 64, 64, (imm), *(dst), (__v64qi)(src1), (__v64qi)(src2)))

/// Rank-4 MX INT8 outer product. A = MX INT8 signed, B = MX INT8 signed.
/// OCP MX block scaling via the BSR, with FP32 accumulation into the
/// destination tile.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TOP4MXBSSPS </c> instruction.
///
/// \param dst
///    Pointer to destination/accumulator __acetile.
/// \param src1
///    The A input: ZMM vector of MX INT8 signed values.
/// \param src2
///    The B input: ZMM vector of MX INT8 signed values.
/// \param imm
///    Scale group selector, formed from \c _MM_ACE_SCALE_A and
///    \c _MM_ACE_SCALE_B; all other bits are reserved and must be zero.
#define __tile_ace_top4mxbssps(dst, src1, src2, imm)                           \
  (*(dst) = __builtin_ia32_top4mxbssps_internal(                               \
       16, 64, 64, (imm), *(dst), (__v64qi)(src1), (__v64qi)(src2)))

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
  *dst = __builtin_ia32_tilemovcolinsert_internal(16, 64, (__v16si)src, idx);
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
  *dst = __builtin_ia32_tilemovrowinsert_internal(16, 64, (__v16si)src, idx);
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

// clang-format on

#undef __DEFAULT_FN_ATTRS_ACE

#endif /* __x86_64__ && __SSE2__ */
#endif /* __ACEV1INTRIN_H */
