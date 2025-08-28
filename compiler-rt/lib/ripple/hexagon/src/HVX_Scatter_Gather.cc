//==============================================================================
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==============================================================================
//
// Part of the Ripple vector library to support the HVX gather and scatter
// instructions.
//
//==============================================================================

#include "lib_func_attrib.h"
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <ripple/zip.h>
#include <ripple_hvx.h>
#include <type_traits>

// _____________________ Readable local macro names ____________________________
//
//

#define HVX_GATHER_32(dst, src, index, region_last_byte)                       \
  Q6_vgather_ARMVw((dst), (uint32_t)(src), (region_last_byte), (index))

#define HVX_GATHER_MASKED_32(dst, src, index, region_last_byte, mask)          \
  Q6_vgather_AQRMVw((dst), Q6_Q_vand_VR((mask), ~0), (uint32_t)(src),          \
                    (region_last_byte), (index))

#define HVX_GATHER_16(dst, src, index, region_last_byte)                       \
  Q6_vgather_ARMVh((dst), (uint32_t)(src), (region_last_byte), (index))

#define HVX_GATHER_MASKED_16(dst, src, index, region_last_byte, mask)          \
  Q6_vgather_AQRMVh((dst), Q6_Q_vand_VR((mask), ~0), (uint32_t)(src),          \
                    (region_last_byte), (index))

#define HVX_GATHER_16_32(dst, src, index, region_last_byte)                    \
  Q6_vgather_ARMWw((dst), (uint32_t)(src), (region_last_byte), (index))

#define HVX_GATHER_MASKED_16_32(dst, src, index, region_last_byte, mask)       \
  Q6_vgather_AQRMWw((dst), Q6_Q_vand_VR((mask), ~0), (uint32_t)(src),          \
                    (region_last_byte), (index))

#define HVX_SCATTER_32(dst, index, src, region_last_byte)                      \
  Q6_vscatter_RMVwV((uint32_t)(dst), (region_last_byte), (index), (src))

#define HVX_SCATTER_MASKED_32(dst, index, src, region_last_byte, mask)         \
  Q6_vscatter_QRMVwV(Q6_Q_vand_VR((mask), ~0), (uint32_t)(dst),                \
                     (region_last_byte), (index), (src))

#define HVX_SCATTER_16(dst, index, src, region_last_byte)                      \
  Q6_vscatter_RMVhV((uint32_t)(dst), (region_last_byte), (index), (src))

#define HVX_SCATTER_MASKED_16(dst, index, src, region_last_byte, mask)         \
  Q6_vscatter_QRMVhV(Q6_Q_vand_VR(mask, ~0), (uint32_t)(dst),                  \
                     (region_last_byte), (index), (src))

#define HVX_SCATTER_16_32(dst, index, src, region_last_byte)                   \
  Q6_vscatter_RMWwV((uint32_t)(dst), (region_last_byte), (index), (src))

#define HVX_SCATTER_MASKED_16_32(dst, index, src, region_last_byte, mask)      \
  Q6_vscatter_QRMWwV(Q6_Q_vand_VR(mask, ~0), (uint32_t)(dst),                  \
                     (region_last_byte), (index), (src))

#define SCATTER_RELEASE(ADDR)                                                  \
  asm volatile("vmem(%0 + #0):scatter_release\n" ::"r"(ADDR));

#if SYNC_VECTOR
#define HVX_SCATTER_SYNC(dst)                                                  \
  {                                                                            \
    /* This dummy load from scatter destination is to complete the             \
     * synchronization. Normally this load would be deferred as long as        \
     * possible to minimize stalls.*/                                          \
    SCATTER_RELEASE(dst);                                                      \
    volatile HVX_Vector vDummy = *(HVX_Vector *)dst;                           \
  }
#else
#define HVX_SCATTER_SYNC(dst)                                                  \
  { SCATTER_RELEASE(dst); }
#endif

#if SYNC_VECTOR
#define HVX_GATHER_SYNC(dst)                                                   \
  {                                                                            \
    /* This dummy read of gather destination will stall until completion */    \
    volatile HVX_Vector vDummy = *(HVX_Vector *)(dst);                         \
  }
#else
#define HVX_GATHER_SYNC(dst)                                                   \
  { /* No-op when SYNC_VECTOR is not defined */                                \
  }
#endif

namespace {

/// Convert a double-vector mask of 32-bit elements to a mask of 16-bit
/// elements. This is needed for double-resource gathers because the input mask
/// has the same shape as the index array.
/// TODO: It would be more correct not just discard high half-words, but
/// generate the Q vector directly by comparing the original masks to zeroes.
/// However, there are no double-vector compares, therefore, we will need to
/// combine two Q vectors.
inline HVX_Vector hvx_pack_mask(HVX_VectorPair mask) {
  return Q6_Vh_vpack_VwVw_sat(Q6_V_hi_W(mask), Q6_V_lo_W(mask));
}

// A wrapper for double-vector shift.
v64i32 Q6_vasl(v64i32 x, unsigned amt) {
  return Q6_W_vcombine_VV(Q6_Vw_vasl_VwR(Q6_V_hi_W(x), amt),
                          Q6_Vw_vasl_VwR(Q6_V_lo_W(x), amt));
}

// Wrappers for type-base shifts.
template <class Elem> HVX_Vector Q6_vlsr(HVX_Vector x, unsigned amt);

template <> HVX_Vector Q6_vlsr<int16_t>(HVX_Vector x, unsigned amt) {
  return Q6_Vuh_vlsr_VuhR(x, amt);
}

template <> HVX_Vector Q6_vlsr<int32_t>(HVX_Vector x, unsigned amt) {
  return Q6_Vuw_vlsr_VuwR(x, amt);
}

// Overloaded versions of HVX intrinsics. These should be used with care as
// vector types of different shapes are mutually convertible.
__attribute__((always_inline)) void Q6_gather(void *dst, const void *src,
                                              size_t last_byte, v64i16 index) {
  HVX_GATHER_16(dst, src, index, last_byte);
}

__attribute__((always_inline)) void Q6_gather(void *dst, const void *src,
                                              size_t last_byte, v64i16 index,
                                              v64i16 mask) {
  HVX_GATHER_MASKED_16(dst, src, index, last_byte, mask);
}

__attribute__((always_inline)) void Q6_gather(void *dst, const void *src,
                                              size_t last_byte, v64i32 index) {
  // The double resource gathers take the even index elements in the first half
  // and the odd ones in the second.
  index = Q6_W_vdeal_VVR(Q6_V_hi_W(index), Q6_V_lo_W(index), 128 - 4);
  HVX_GATHER_16_32(dst, src, index, last_byte);
}

__attribute__((always_inline)) void Q6_gather(void *dst, const void *src,
                                              size_t last_byte, v64i32 index,
                                              v64i16 mask) {
  index = Q6_W_vdeal_VVR(Q6_V_hi_W(index), Q6_V_lo_W(index), 128 - 4);
  HVX_GATHER_MASKED_16_32(dst, src, index, last_byte, mask);
}

__attribute__((always_inline)) void Q6_scatter(void *dst, size_t last_byte,
                                               HVX_Vector src, v64i16 offsets) {
  HVX_SCATTER_16(dst, offsets, src, last_byte);
}

__attribute__((always_inline)) void Q6_scatter(void *dst, size_t last_byte,
                                               HVX_Vector src, v64i16 offsets,
                                               v64i16 mask) {
  HVX_SCATTER_MASKED_16(dst, offsets, src, last_byte, mask);
}

__attribute__((always_inline)) void Q6_scatter(void *dst, size_t last_byte,
                                               HVX_Vector src, v64i32 offsets) {
  // For double resource scatter, instead of reshuffling the indexes, which
  // takes two vectors, adjust the source, which is one vector only.
  v64i16 adjusted_src = Q6_Vh_vshuff_Vh(src);
  HVX_SCATTER_16_32(dst, offsets, adjusted_src, last_byte);
}

__attribute__((always_inline)) void Q6_scatter(void *dst, size_t last_byte,
                                               HVX_Vector src, v64i32 offsets,
                                               v64i16 mask) {
  v64i32 adjusted_offsets =
      Q6_W_vdeal_VVR(Q6_V_hi_W(offsets), Q6_V_lo_W(offsets), -4);
  HVX_SCATTER_MASKED_16_32(dst, adjusted_offsets, src, last_byte, mask);
}

__attribute__((always_inline)) void Q6_scatter(void *dst, size_t last_byte,
                                               HVX_Vector src, v32i32 offsets) {
  HVX_SCATTER_32(dst, offsets, src, last_byte);
}

__attribute__((always_inline)) void Q6_scatter(void *dst, size_t last_byte,
                                               HVX_Vector src, v32i32 index,
                                               v32i32 mask) {
  HVX_SCATTER_MASKED_32(dst, index, src, last_byte, mask);
}

} // namespace

// Template version of ripple_to_hvx. It should eventually go to ripple_hvx.h.
template <typename T, size_t N_EL>
[[gnu::always_inline]] T __attribute__((vector_size(sizeof(T) * N_EL)))
ripple_to_hvx_gen(ripple_block_t BS, T x) {
  T tmp[N_EL];
  tmp[ripple_id(BS, 0)] = x;
  return *((T __attribute__((vector_size(sizeof(T) * N_EL))) *)tmp);
}

using namespace rzip;

extern "C" {

// vgather and vscatter aren't elementwise,
// in that we can't implement gather(dst, src, v64i32 index)
// as gather(dst, src, index.lo); gather(dst, src, index.hi)

/// @brief source function to duplicate the low half as pairs
static size_t pairs_from_lo(size_t dst_idx, size_t block_size) {
  return dst_idx / 2;
}

/// @brief source function to duplicate the low half as pairs
static size_t pairs_from_hi(size_t dst_idx, size_t block_size) {
  size_t h = block_size / 2;
  return h + dst_idx / 2;
}

// _______________________________ vgather _____________________________________
//
//

// _General principle_: Actual gather implementation is provided for ints.
//                      A wrapper is created for floats and unsigned inputs.

/// @brief vgather 64-bit vector using 32 32-bit indices (unmasked)
/// @param dst destination base address. Must be aligned on 128 bytes.
/// @param src source base address.
/// @param index offset indices (from src) of the gather
/// @param region_size number of 64-bit elements in the gather region (from src)
RIPPLE_INTRIN_INLINE void ripple_hvx_gather_i64(int64_t *dst,
                                                const int64_t *src,
                                                v32i32 index,
                                                size_t region_size) {
  auto BS = ripple_set_block_shape(0, 32);
  size_t v = ripple_id(BS, 0);
  int32_t idx = hvx_to_ripple(BS, 32, i32, index);
  // Initiate two 32-bit gathers, each with [index, index+4]
  // pairs, for lo and hi
  int32_t idx_lo = ripple_shuffle(idx, pairs_from_lo) * 2;
  int32_t idx_hi = ripple_shuffle(idx, pairs_from_hi) * 2;
  idx_lo |= v & 1;
  idx_hi |= v & 1;
  v32i32 index_lo = ripple_to_hvx(BS, 32, i32, idx_lo * sizeof(uint32_t));
  v32i32 index_hi = ripple_to_hvx(BS, 32, i32, idx_hi * sizeof(uint32_t));
  int32_t region_last_byte = (region_size << 3) - 1;
  HVX_GATHER_32(dst, src, index_lo, region_last_byte);
  HVX_GATHER_32(dst + 16, src, index_hi, region_last_byte);
}

/// @brief vgather 64-bit vector using 32 32-bit indices (masked)
/// @param dst destination base address. Must be aligned on 128 bytes.
/// @param src source base address.
/// @param index offset indices (from src) of the gather
/// @param region_size number of 64-bit elements in the gather region (from src)
/// @param mask an elementwise mask applied to the gather operation.
RIPPLE_INTRIN_INLINE void
ripple_mask_hvx_gather_i64(int64_t *dst, const int64_t *src, v32i32 index,
                           size_t region_size, v32i32 mask) {
  auto BS = ripple_set_block_shape(0, 32);
  size_t v = ripple_id(BS, 0);
  int32_t idx = hvx_to_ripple(BS, 32, i32, index);
  int32_t msk = hvx_to_ripple(BS, 32, i32, mask);
  // Initiate two 32-bit gathers, each with [index, index+4]
  // pairs, for lo and hi
  int32_t idx_lo = ripple_shuffle(idx, pairs_from_lo) * 2;
  int32_t idx_hi = ripple_shuffle(idx, pairs_from_hi) * 2;
  int32_t msk_lo = ripple_shuffle(msk, pairs_from_lo);
  int32_t msk_hi = ripple_shuffle(msk, pairs_from_hi);
  idx_lo |= v & 1;
  idx_hi |= v & 1;
  int32_t region_last_byte = (region_size << 3) - 1;
  v32i32 index_lo = ripple_to_hvx(BS, 32, i32, idx_lo * sizeof(uint32_t));
  v32i32 mask_lo = ripple_to_hvx(BS, 32, i32, msk_lo ? -1 : 0);
  HVX_GATHER_MASKED_32(dst, src, index_lo, region_last_byte, mask_lo);
  v32i32 index_hi = ripple_to_hvx(BS, 32, i32, idx_hi * sizeof(uint32_t));
  v32i32 mask_hi = ripple_to_hvx(BS, 32, i32, msk_hi ? -1 : 0);
  HVX_GATHER_MASKED_32(dst + 16, src, index_hi, region_last_byte, mask_hi);
}

/// @brief vgather 32-bit vector using 32-bit indices (unmasked)
/// @param dst destination base address. Must be aligned on 128 bytes.
/// @param src source base address.
/// @param index offset indices (from src) of the gather
/// @param region_size number of 32-bit elements in the gather region (from src)
RIPPLE_INTRIN_INLINE void ripple_hvx_gather_i32(int32_t *dst,
                                                const int32_t *src,
                                                v32i32 index,
                                                size_t region_size) {
  int32_t region_last_byte = (region_size << 2) - 1;
  HVX_GATHER_32(dst, src, Q6_Vw_vasl_VwR(index, 2), region_last_byte);
}

/// @brief vgather 32-bit vector using 32-bit indices (masked)
/// @param dst destination base address. Must be aligned on 128 bytes.
/// @param src source base address.
/// @param index offset indices (from src) of the gather
/// @param region_size number of 32-bit elements in the gather region (from src)
/// @param mask an elementwise mask applied to the gather operation.
RIPPLE_INTRIN_INLINE void
ripple_mask_hvx_gather_i32(int32_t *dst, const int32_t *src, v32i32 index,
                           size_t region_size, v32i32 mask) {
  int32_t region_last_byte = (region_size << 2) - 1;
  HVX_GATHER_MASKED_32(dst, src, Q6_Vw_vasl_VwR(index, 2), region_last_byte,
                       Q6_Q_vcmp_gt_VuwVuw(mask, Q6_V_vsplat_R(0)));
}

/// @brief vgather 16-bit vector using 32-bit indices (unmasked)
/// @param dst destination base address. Must be aligned on 128 bytes.
/// @param src source base address.
/// @param index offset indices (from src) of the gather
/// @param region_size number of 16-bit elements in the gather region (from src)
RIPPLE_INTRIN_INLINE void ripple_hvx_gather_i16(int16_t *dst,
                                                const int16_t *src,
                                                v64i32 index,
                                                size_t region_size) {
  int32_t region_last_byte = (region_size << 1) - 1;
  Q6_gather(dst, src, region_last_byte, Q6_vasl(index, 1));
  HVX_GATHER_SYNC(dst);
}

/// @brief vgather 16-bit vector using 32-bit indices (masked)
/// @param dst destination base address. Must be aligned on 128 bytes.
/// @param src source base address.
/// @param index offset indices (from src) of the gather
/// @param region_size number of 16-bit elements in the gather region (from src)
/// @param mask an elementwise mask applied to the gather operation.
RIPPLE_INTRIN_INLINE void
ripple_mask_hvx_gather_i16(int16_t *dst, const int16_t *src, v64i32 index,
                           size_t region_size, v64i32 mask) {
  int32_t region_last_byte = (region_size << 1) - 1;
  Q6_gather(dst, src, region_last_byte, Q6_vasl(index, 1),
            Q6_Q_vcmp_gt_VuhVuh(hvx_pack_mask(mask), Q6_Vh_vsplat_R(0)));
  HVX_GATHER_SYNC(dst);
}
}

namespace {

/// A generic function to implement a sub-element size gather with masked
/// gather on elements of a larger size. An element is broken down to
/// M "sub-elements" S(M-1) ... S(1) S(0), where M is typically two, and a slice
/// is an array of those sub-elements. One invocation of this function processes
/// one staggered array of sub-elements ("slice").
/// @param Slice sub-element index, must be constexpr because of ripple_slice
/// @param N array size
/// @param T sub-element type
/// @param ElementT element type
/// @param IndexT index type
template <unsigned Slice, size_t N, typename T, typename ElementT,
          typename IndexT>
[[gnu::always_inline]] void sub_gather(ripple_block_t BS, T *dst, const T *src,
                                       size_t last_byte, IndexT index,
                                       int mask) {
  auto sliced_index = ripple_slice(index, -1, Slice);
  auto sliced_mask = ripple_slice(int32_t(mask), -1, Slice);
  auto bitmask = typename std::make_unsigned<T>::type(-1)
                 << (sizeof(T) * 8 * Slice);
  if (sliced_mask == 0)
    bitmask = 0;
  Q6_gather(dst, src, last_byte,
            ripple_to_hvx_gen<IndexT, N>(BS, sliced_index - Slice),
            ripple_to_hvx_gen<ElementT, N>(BS, bitmask));
}

// Fill the elements which index is zero and which are not in the lowest
// position. Since gather offsets were reduced by Slice, the offsets of those
// elements become negative and the elements will be ignored by the normal
// gather. The current implmentation only supports the odd half of the two
// sub-element case.
template <size_t N, typename T, typename ElementT, typename IndexT>
[[gnu::always_inline]] void sub_gather_zero(ripple_block_t BS, T *dst,
                                            const T *src, size_t last_byte,
                                            IndexT index, int mask) {
  constexpr unsigned Slice = 1;
  auto sliced_index = ripple_slice(index, -1, Slice);
  auto sliced_mask = ripple_slice(int32_t(mask), -1, Slice);
  auto bitmask = typename std::make_unsigned<T>::type(-1)
                 << (sizeof(T) * 8 * Slice);
  ElementT zero_element = src[0] << (sizeof(T) * 8 * Slice);
  if (sliced_index != 0 || sliced_mask == 0)
    bitmask = 0;
  // Use a masked store to fill elements with zero indexes. There are
  // potential problems with this:
  // - It is not documented if interleaving of gather and normal stores
  // produces the expected result.
  // - A masked store is vector-aligned, which requires the destination to be
  // vector-aligned too. These problems can be avoided with unaligned load,
  // update, and store cycle.
  auto hvx_mask = ripple_to_hvx_gen<ElementT, N>(BS, bitmask);
  auto hvx_value = ripple_to_hvx_gen<ElementT, N>(BS, zero_element);
  Q6_vmem_QRIV(hvx_mask, dst, hvx_value);
}

// A separate implementation of sub-element non-masked gathers, similar to the
// masked one above but without masks.
template <unsigned Slice, size_t N, typename T, typename ElementT,
          typename IndexT>
[[gnu::always_inline]] void sub_gather(ripple_block_t BS, T *dst, const T *src,
                                       size_t last_byte, IndexT index) {
  auto sliced_index = ripple_slice(index, -1, Slice);
  auto bitmask = typename std::make_unsigned<T>::type(-1U)
                 << (sizeof(T) * 8 * Slice);
  Q6_gather(dst, src, last_byte,
            ripple_to_hvx_gen<IndexT, N>(BS, sliced_index - Slice),
            ripple_to_hvx_gen<ElementT, N>(BS, bitmask));
}

template <size_t N, typename T, typename ElementT, typename IndexT>
[[gnu::always_inline]] void sub_gather_zero(ripple_block_t BS, T *dst,
                                            const T *src, size_t last_byte,
                                            IndexT index) {
  constexpr unsigned Slice = 1;
  auto sliced_index = ripple_slice(index, -1, Slice);
  auto bitmask = typename std::make_unsigned<T>::type(-1U)
                 << (sizeof(T) * 8 * Slice);
  ElementT zero_element = src[0] << (sizeof(T) * 8 * Slice);
  if (sliced_index != 0)
    bitmask = 0;
  auto hvx_mask = ripple_to_hvx_gen<ElementT, N>(BS, bitmask);
  auto hvx_value = ripple_to_hvx_gen<ElementT, N>(BS, zero_element);
  Q6_vmem_QRIV(hvx_mask, dst, hvx_value);
}

} // namespace

extern "C" {

/// @brief vgather 8-bit vector using 32-bit indices (unmasked)
/// This is boils down to 4 transfers.
/// @param dst destination base address. Must be aligned on 128 bytes.
/// @param src source base address.
/// @param index offset indices (from src) of the gather
/// @param region_size number of bytes in the gather region (from src)
RIPPLE_INTRIN_INLINE void ripple_hvx_gather_i8(int8_t *dst, const int8_t *src,
                                               v128i32 index,
                                               size_t region_size) {
  auto BS = ripple_set_block_shape(0, 64, 2);
  size_t last_byte = region_size - 1;
  auto idx = hvx_to_ripple_2d(BS, 128, i32, index);
  idx = ripple_shuffle(idx, shuffle_unzip<2, 0, 0>);
  sub_gather<0, 64, int8_t, int16_t, int32_t>(BS, dst, src, last_byte, idx);
  sub_gather<1, 64, int8_t, int16_t, int32_t>(BS, dst, src, last_byte, idx);
  sub_gather_zero<64, int8_t, int16_t, int32_t>(BS, dst, src, last_byte, idx);
}

/// @brief vgather 8-bit vector using 32-bit indices (masked)
/// @param dst destination base address. Must be aligned on 128 bytes.
/// @param src source base address.
/// @param index offset indices (from src) of the gather
/// @param region_size number of bytes in the gather region (from src)
/// @param mask an elementwise mask applied to the gather operation.
RIPPLE_INTRIN_INLINE void
ripple_mask_hvx_gather_i8(int8_t *dst, const int8_t *src, v128i32 index,
                          size_t region_size, v128i32 mask) {
  auto BS = ripple_set_block_shape(0, 64, 2);
  size_t last_byte = region_size - 1;
  auto idx = hvx_to_ripple_2d(BS, 128, i32, index);
  idx = ripple_shuffle(idx, shuffle_unzip<2, 0, 0>);
  auto msk = hvx_to_ripple_2d(BS, 128, i32, mask);
  msk = ripple_shuffle(msk, shuffle_unzip<2, 0, 0>);
  sub_gather<0, 64, int8_t, int16_t, int32_t>(BS, dst, src, last_byte, idx,
                                              msk);
  sub_gather<1, 64, int8_t, int16_t, int32_t>(BS, dst, src, last_byte, idx,
                                              msk);
  sub_gather_zero<64, int8_t, int16_t, int32_t>(BS, dst, src, last_byte, idx,
                                                msk);
}

/// @brief vgather 16-bit vector using 16-bit indices (unmasked)
/// @param dst destination base address. Must be aligned on 128 bytes.
/// @param src source base address.
/// @param index offset indices (from src) of the gather
/// @param region_size number of 16-bit elements in the gather region (from src)
RIPPLE_INTRIN_INLINE void ripple_hvx_gather_i16_16(int16_t *dst,
                                                   const int16_t *src,
                                                   v64i16 index,
                                                   size_t region_size) {
  int32_t region_last_byte = (region_size << 1) - 1;
  HVX_GATHER_16(dst, src, Q6_Vh_vasl_VhR(index, 1), region_last_byte);
}

/// @brief vgather 16-bit vector using 16-bit indices (masked)
/// @param dst destination base address. Must be aligned on 128 bytes.
/// @param src source base address.
/// @param index offset indices (from src) of the gather
/// @param region_size number of 16-bit elements in the gather region (from src)
/// @param mask an elementwise mask applied to the gather operation.
RIPPLE_INTRIN_INLINE void
ripple_mask_hvx_gather_i16_16(int16_t *dst, const int16_t *src, v64i16 index,
                              size_t region_size, v64i16 mask) {
  int32_t region_last_byte = (region_size << 1) - 1;
  HVX_GATHER_MASKED_16(dst, src, Q6_Vh_vasl_VhR(index, 1), region_last_byte,
                       Q6_Q_vcmp_gt_VuhVuh(mask, Q6_Vh_vsplat_R(0)));
}

/// @brief vgather 8-bit vector using 16-bit indices (unmasked)
/// @param dst destination base address. Must be aligned on 128 bytes.
/// @param src source base address.
/// @param index offset indices (from src) of the gather
/// @param region_size number of bytes in the gather region (from src)
RIPPLE_INTRIN_INLINE void ripple_hvx_gather_i8_16(int8_t *dst,
                                                  const int8_t *src,
                                                  v128i16 index,
                                                  size_t region_size) {
  auto BS = ripple_set_block_shape(0, 64, 2);
  size_t last_byte = region_size - 1;
  auto idx = hvx_to_ripple_2d(BS, 128, i16, index);
  idx = ripple_shuffle(idx, shuffle_unzip<2, 0, 0>);
  sub_gather<0, 64, int8_t, int16_t, int16_t>(BS, dst, src, last_byte, idx);
  sub_gather<1, 64, int8_t, int16_t, int16_t>(BS, dst, src, last_byte, idx);
  sub_gather_zero<64, int8_t, int16_t, int16_t>(BS, dst, src, last_byte, idx);
}

/// @brief vgather 8-bit vector using 16-bit indices (masked)
/// @param dst destination base address. Must be aligned on 128 bytes.
/// @param src source base address.
/// @param index offset indices (from src) of the gather
/// @param region_size number of bytes in the gather region (from src)
/// @param mask an elementwise mask applied to the gather operation.
RIPPLE_INTRIN_INLINE void
ripple_mask_hvx_gather_i8_16(int8_t *dst, const int8_t *src, v128i16 index,
                             size_t region_size, v128i16 mask) {
  auto BS = ripple_set_block_shape(0, 64, 2);
  size_t last_byte = region_size - 1;
  auto idx = hvx_to_ripple_2d(BS, 128, i16, index);
  idx = ripple_shuffle(idx, shuffle_unzip<2, 0, 0>);
  auto msk = hvx_to_ripple_2d(BS, 128, i16, mask);
  msk = ripple_shuffle(msk, shuffle_unzip<2, 0, 0>);
  sub_gather<0, 64, int8_t, int16_t, int16_t>(BS, dst, src, last_byte, idx,
                                              msk);
  sub_gather<1, 64, int8_t, int16_t, int16_t>(BS, dst, src, last_byte, idx,
                                              msk);
  sub_gather_zero<64, int8_t, int16_t, int16_t>(BS, dst, src, last_byte, idx,
                                                msk);
}
}

namespace {

template <unsigned Slice, size_t N, typename ElementT, typename IndexT,
          typename T>
[[gnu::always_inline]] void partial_scatter(ripple_block_t BS, T *dst,
                                            size_t last_byte, ElementT src,
                                            IndexT idx) {
  ElementT src_slice = ripple_slice(src, -1, Slice);
  int offset = idx * sizeof(T) + Slice * sizeof(ElementT);
  Q6_scatter(dst, last_byte, ripple_to_hvx_gen<ElementT, N>(BS, src_slice),
             ripple_to_hvx_gen<IndexT, N>(BS, offset));
}

template <unsigned Slice, size_t N, typename ElementT, typename IndexT,
          typename MaskT, typename T>
[[gnu::always_inline]] void partial_scatter(ripple_block_t BS, T *dst,
                                            size_t last_byte, ElementT src,
                                            IndexT idx, MaskT msk) {
  ElementT src_slice = ripple_slice(src, -1, Slice);
  int offset = idx * sizeof(T) + Slice * sizeof(ElementT);
  Q6_scatter(dst, last_byte, ripple_to_hvx_gen<ElementT, N>(BS, src_slice),
             ripple_to_hvx_gen<IndexT, N>(BS, offset),
             ripple_to_hvx_gen<MaskT, N>(BS, msk != 0 ? -1 : 0));
}

} // namespace

extern "C" {

// ______________________________ vscatter _____________________________________
//
//

// _General principle_: Actual scatter implementation is provided for ints.
//                      A wrapper is created for floats and unsigned inputs.

/// @brief vscatter 64-bit vector using 32 32-bit indices (unmasked)
/// @param dst destination base address. Must be aligned on 128 bytes.
/// @param src source base address.
/// @param index offset indices (from src) of the gather
/// @param region_size number of 64-bit elements in the gather region (from src)
RIPPLE_INTRIN_INLINE void ripple_hvx_scatter_i64(int64_t *dst, v32i32 index,
                                                 v32i64 src,
                                                 size_t region_size) {
  size_t last_byte = region_size * sizeof(int64_t) - 1;
  // Treat the source vector as a twice longer vector of smaller elements.
  auto BS = ripple_set_block_shape(0, 32, 2);
  auto s = hvx_to_ripple_2d(BS, 64, i32, src);
  s = ripple_shuffle(s, shuffle_unzip<2, 0, 0>);
  auto BS2 = ripple_set_block_shape(0, 32);
  auto idx = hvx_to_ripple(BS2, 32, i32, index);
  partial_scatter<0, 32, int32_t, int32_t>(BS2, dst, last_byte, s, idx);
  partial_scatter<1, 32, int32_t, int32_t>(BS2, dst, last_byte, s, idx);
}

/// @brief vscatter 64-bit vector using 32 32-bit indices (masked)
/// @param dst destination base address. Must be aligned on 128 bytes.
/// @param src source base address.
/// @param index offset indices (from src) of the gather
/// @param region_size number of 64-bit elements in the gather region (from src)
/// @param mask an elementwise mask applied to the gather operation.
RIPPLE_INTRIN_INLINE void ripple_mask_hvx_scatter_i64(int64_t *dst,
                                                      v32i32 index, v32i64 src,
                                                      size_t region_size,
                                                      v32i32 mask) {
  size_t last_byte = region_size * sizeof(int64_t) - 1;
  // Treat the source vector as a twice longer vector of smaller elements.
  auto BS = ripple_set_block_shape(0, 32, 2);
  auto s = hvx_to_ripple_2d(BS, 64, i32, src);
  s = ripple_shuffle(s, shuffle_unzip<2, 0, 0>);
  auto BS2 = ripple_set_block_shape(0, 32);
  auto idx = hvx_to_ripple(BS2, 32, i32, index);
  auto msk = hvx_to_ripple(BS2, 32, i32, mask);
  partial_scatter<0, 32, int32_t, int32_t, int32_t>(BS2, dst, last_byte, s, idx,
                                                    msk);
  partial_scatter<1, 32, int32_t, int32_t, int32_t>(BS2, dst, last_byte, s, idx,
                                                    msk);
}

/// @brief vscatter 32-bit vector using 32-bit indices (unmasked)
/// @param dst destination base address. Must be aligned on 128 bytes.
/// @param src source base address.
/// @param index offset indices (from src) of the gather
/// @param region_size number of 32-bit elements in the gather region (from src)
RIPPLE_INTRIN_INLINE void ripple_hvx_scatter_i32(int32_t *dst, v32i32 index,
                                                 v32i32 src,
                                                 size_t region_size) {
  int32_t region_last_byte = (region_size << 2) - 1;
  HVX_SCATTER_32(dst, Q6_Vw_vasl_VwR(index, 2), src, region_last_byte);
}

/// @brief vscatter 32-bit vector using 32-bit indices (masked)
/// @param dst destination base address. Must be aligned on 128 bytes.
/// @param src source base address.
/// @param index offset indices (from src) of the gather
/// @param region_size number of 32-bit elements in the gather region (from src)
/// @param mask an elementwise mask applied to the gather operation.
RIPPLE_INTRIN_INLINE void ripple_mask_hvx_scatter_i32(int32_t *dst,
                                                      v32i32 index, v32i32 src,
                                                      size_t region_size,
                                                      v32i32 mask) {
  int32_t region_last_byte = (region_size << 2) - 1;
  HVX_SCATTER_MASKED_32(dst, Q6_Vw_vasl_VwR(index, 2), src, region_last_byte,
                        Q6_Q_vcmp_gt_VuwVuw(mask, Q6_V_vsplat_R(0)));
}

/// @brief vscatter 16-bit vector using 32-bit indices (unmasked)
/// @param dst destination base address. Must be aligned on 128 bytes.
/// @param src source base address.
/// @param index offset indices (from src) of the gather
/// @param region_size number of 16-bit elements in the gather region (from src)
RIPPLE_INTRIN_INLINE void ripple_hvx_scatter_i16(int16_t *dst, v64i32 index,
                                                 v64i16 src,
                                                 size_t region_size) {
  size_t region_last_byte = (region_size << 1) - 1;
  v64i32 offsets = Q6_vasl(index, 1);
  Q6_scatter(dst, region_last_byte, src, offsets);
  HVX_SCATTER_SYNC(dst);
}

/// @brief vscatter 8-bit vector using 32-bit indices (masked)
/// @param dst destination base address. Must be aligned on 128 bytes.
/// @param src source base address.
/// @param index offset indices (from src) of the gather
/// @param region_size number of bytes in the gather region (from src)
/// @param mask an elementwise mask applied to the gather operation.
RIPPLE_INTRIN_INLINE void ripple_mask_hvx_scatter_i16(int16_t *dst,
                                                      v64i32 index, v64i16 src,
                                                      size_t region_size,
                                                      v64i32 mask) {
  int32_t region_last_byte = (region_size << 1) - 1;
  v64i32 offsets = Q6_vasl(index, 1);
  Q6_scatter(dst, region_last_byte, src, offsets,
             Q6_Q_vcmp_gt_VuhVuh(hvx_pack_mask(mask), Q6_Vh_vsplat_R(0)));
  HVX_SCATTER_SYNC(dst);
}
}

namespace {

template <unsigned Slice, size_t N, typename ElementT, typename IndexT,
          typename T>
[[gnu::always_inline]] void sub_scatter(ripple_block_t BS, T *dst,
                                        size_t last_byte, HVX_Vector src,
                                        IndexT index) {
  auto sliced_index = ripple_slice(index, -1, Slice);
  IndexT offset = (sliced_index - Slice) * sizeof(T);
  auto bitmask = typename std::make_unsigned<T>::type(-1U)
                 << (sizeof(T) * 8 * Slice);
  Q6_scatter(dst, last_byte, src, ripple_to_hvx_gen<IndexT, N>(BS, offset),
             ripple_to_hvx_gen<ElementT, N>(BS, bitmask));
}

// Write any odd sub-elements with zero index by shifting them into even
// positions. These sub-elements are not written by the previous scatter
// instruction because their adjusted offset becomes negative. This
// implementation only works for two sub-elements.
template <size_t N, typename ElementT, typename IndexT, typename T>
[[gnu::always_inline]] void sub_scatter_zero(ripple_block_t BS, T *dst,
                                             size_t last_byte, HVX_Vector src,
                                             IndexT index) {
  constexpr unsigned Slice = 1;
  auto sliced_index = ripple_slice(index, -1, Slice);
  auto bitmask = typename std::make_unsigned<T>::type(-1U)
                 << (sizeof(T) * 8 * Slice);
  if (sliced_index != 0)
    bitmask = 0;
  Q6_scatter(dst, last_byte, Q6_vlsr<ElementT>(src, sizeof(T) * 8),
             Q6_V_vsplat_R(0),
             Q6_vlsr<ElementT>(ripple_to_hvx_gen<ElementT, N>(BS, bitmask),
                               sizeof(T) * 8));
}

template <unsigned Slice, size_t N, typename ElementT, typename IndexT,
          typename T>
[[gnu::always_inline]] void sub_scatter(ripple_block_t BS, T *dst,
                                        size_t last_byte, HVX_Vector src,
                                        IndexT index, int mask) {
  auto sliced_index = ripple_slice(index, -1, Slice);
  auto sliced_mask = ripple_slice(int32_t(mask), -1, Slice);
  IndexT offset = (sliced_index - Slice) * sizeof(T);
  auto bitmask = typename std::make_unsigned<T>::type(-1U)
                 << (sizeof(T) * 8 * Slice);
  if (sliced_mask == 0)
    bitmask = 0;
  Q6_scatter(dst, last_byte, src, ripple_to_hvx_gen<IndexT, N>(BS, offset),
             ripple_to_hvx_gen<ElementT, N>(BS, bitmask));
}

template <size_t N, typename ElementT, typename IndexT, typename T>
[[gnu::always_inline]] void sub_scatter_zero(ripple_block_t BS, T *dst,
                                             size_t last_byte, HVX_Vector src,
                                             IndexT index, int mask) {
  constexpr unsigned Slice = 1;
  auto sliced_index = ripple_slice(index, -1, Slice);
  auto sliced_mask = ripple_slice(int32_t(mask), -1, Slice);
  auto bitmask = typename std::make_unsigned<T>::type(-1U)
                 << (sizeof(T) * 8 * Slice);
  if (sliced_index != 0 || sliced_mask == 0)
    bitmask = 0;
  Q6_scatter(dst, last_byte, Q6_vlsr<ElementT>(src, sizeof(T) * 8),
             Q6_V_vsplat_R(0),
             Q6_vlsr<ElementT>(ripple_to_hvx_gen<ElementT, N>(BS, bitmask),
                               sizeof(T) * 8));
}

} // namespace

extern "C" {

// @brief vscatter 8-bit vector using 32-bit indices (unmasked)
/// This is boils down to 4 transfers.
/// @param dst destination base address. Must be aligned on 128 bytes.
/// @param src source base address.
/// @param index offset indices (from src) of the gather
/// @param region_size number of bytes in the gather region (from src)
RIPPLE_INTRIN_INLINE void ripple_hvx_scatter_i8(int8_t *dst, v128i32 index,
                                                v128i8 src,
                                                size_t region_size) {
  auto BS = ripple_set_block_shape(0, 64, 2);
  size_t last_byte = region_size - 1;
  auto idx = hvx_to_ripple_2d(BS, 128, i32, index);
  idx = ripple_shuffle(idx, shuffle_unzip<2, 0, 0>);
  sub_scatter<0, 64, int16_t, int32_t>(BS, dst, last_byte, src, idx);
  sub_scatter<1, 64, int16_t, int32_t>(BS, dst, last_byte, src, idx);
  sub_scatter_zero<64, int16_t, int32_t>(BS, dst, last_byte, src, idx);
}

/// @brief vscatter 8-bit vector using 32-bit indices (masked)
/// @param dst destination base address. Must be aligned on 128 bytes.
/// @param src source base address.
/// @param index offset indices (from src) of the gather
/// @param region_size number of bytes in the gather region (from src)
/// @param mask an elementwise mask applied to the gather operation.
RIPPLE_INTRIN_INLINE void ripple_mask_hvx_scatter_i8(int8_t *dst, v128i32 index,
                                                     v128i8 src,
                                                     size_t region_size,
                                                     v128i32 mask) {
  auto BS = ripple_set_block_shape(0, 64, 2);
  size_t last_byte = region_size - 1;
  auto idx = hvx_to_ripple_2d(BS, 128, i32, index);
  idx = ripple_shuffle(idx, shuffle_unzip<2, 0, 0>);
  auto msk = hvx_to_ripple_2d(BS, 128, i32, mask);
  msk = ripple_shuffle(msk, shuffle_unzip<2, 0, 0>);
  sub_scatter<0, 64, int16_t, int32_t>(BS, dst, last_byte, src, idx, msk);
  sub_scatter<1, 64, int16_t, int32_t>(BS, dst, last_byte, src, idx, msk);
  sub_scatter_zero<64, int16_t, int32_t>(BS, dst, last_byte, src, idx, msk);
}

/// @brief vscatter 16-bit vector using 16-bit indices (unmasked)
/// @param dst destination base address. Must be aligned on 128 bytes.
/// @param src source base address.
/// @param index offset indices (from src) of the gather
/// @param region_size number of 16-bit elements in the gather region (from src)
RIPPLE_INTRIN_INLINE void ripple_hvx_scatter_i16_16(int16_t *dst, v64i16 index,
                                                    v64i16 src,
                                                    size_t region_size) {
  int32_t region_last_byte = (region_size << 1) - 1;
  HVX_SCATTER_16(dst, Q6_Vh_vasl_VhR(index, 1), src, region_last_byte);
}

/// @brief vscatter 16-bit vector using 16-bit indices (masked)
/// @param dst destination base address. Must be aligned on 128 bytes.
/// @param src source base address.
/// @param index offset indices (from src) of the gather
/// @param region_size number of 16-bit elements in the gather region (from src)
/// @param mask an elementwise mask applied to the gather operation.
RIPPLE_INTRIN_INLINE void
ripple_mask_hvx_scatter_i16_16(int16_t *dst, v64i16 index, v64i16 src,
                               size_t region_size, v64i16 mask) {
  int32_t region_last_byte = (region_size << 1) - 1;
  HVX_SCATTER_MASKED_16(dst, Q6_Vh_vasl_VhR(index, 1), src, region_last_byte,
                        Q6_Q_vcmp_gt_VuhVuh(mask, Q6_V_vsplat_R(0)));
}

/// @brief vscatter 8-bit vector using 16-bit indices (unmasked)
/// @param dst destination base address. Must be aligned on 128 bytes.
/// @param src source base address.
/// @param index offset indices (from src) of the gather
/// @param region_size number of bytes in the gather region (from src)
RIPPLE_INTRIN_INLINE void ripple_hvx_scatter_i8_16(int8_t *dst, v128i16 index,
                                                   v128i8 src,
                                                   size_t region_size) {
  auto BS = ripple_set_block_shape(0, 64, 2);
  size_t last_byte = region_size - 1;
  auto idx = hvx_to_ripple_2d(BS, 128, i16, index);
  idx = ripple_shuffle(idx, shuffle_unzip<2, 0, 0>);
  sub_scatter<0, 64, int16_t, int16_t>(BS, dst, last_byte, src, idx);
  sub_scatter<1, 64, int16_t, int16_t>(BS, dst, last_byte, src, idx);
  sub_scatter_zero<64, int16_t, int16_t>(BS, dst, last_byte, src, idx);
}

/// @brief vscatter 8-bit vector using 16-bit indices (masked)
/// @param dst destination base address. Must be aligned on 128 bytes.
/// @param src source .
/// @param index offset indices (from src) of the gather
/// @param region_size number of bytes in the gather region (from src)
/// @param mask an elementwise mask applied to the gather operation.
RIPPLE_INTRIN_INLINE void
ripple_mask_hvx_scatter_i8_16(int8_t *dst, v128i16 index, v128i8 src,
                              size_t region_size, v128i16 mask) {
  auto BS = ripple_set_block_shape(0, 64, 2);
  size_t last_byte = region_size - 1;
  auto idx = hvx_to_ripple_2d(BS, 128, i16, index);
  idx = ripple_shuffle(idx, shuffle_unzip<2, 0, 0>);
  auto msk = hvx_to_ripple_2d(BS, 128, i16, mask);
  msk = ripple_shuffle(msk, shuffle_unzip<2, 0, 0>);
  sub_scatter<0, 64, int16_t, int16_t>(BS, dst, last_byte, src, idx, msk);
  sub_scatter<1, 64, int16_t, int16_t>(BS, dst, last_byte, src, idx, msk);
  sub_scatter_zero<64, int16_t, int16_t>(BS, dst, last_byte, src, idx, msk);
}

// _________________________ Float and Unsigned APIs ___________________________
//
//

// ------------------------------ gather ---------------------------------------

/// Declares a typed scatter variant that maps to its int implementation
#define _decl_float_gather(CT, T, W, N)                                        \
  RIPPLE_INTRIN_INLINE void ripple_hvx_gather_##T##W(                          \
      CT *dst, const CT *src, v##N##i32 index, size_t region_size) {           \
    ripple_hvx_gather_i##W((int##W##_t *)dst, (int##W##_t *)src, index,        \
                           region_size);                                       \
  }                                                                            \
  RIPPLE_INTRIN_INLINE void ripple_mask_hvx_gather_##T##W(                     \
      CT *dst, const CT *src, v##N##i32 index, size_t region_size,             \
      v##N##i32 mask) {                                                        \
    ripple_mask_hvx_gather_i##W((int##W##_t *)dst, (int##W##_t *)src, index,   \
                                region_size, mask);                            \
  }

#define _decl_float_gather_16(CT, T, W, N)                                     \
  RIPPLE_INTRIN_INLINE void ripple_hvx_gather_##T##W##_16(                     \
      CT *dst, const CT *src, v##N##i16 index, size_t region_size) {           \
    ripple_hvx_gather_i##W##_16((int##W##_t *)dst, (int##W##_t *)src, index,   \
                                region_size);                                  \
  }                                                                            \
  RIPPLE_INTRIN_INLINE void ripple_mask_hvx_gather_##T##W##_16(                \
      CT *dst, const CT *src, v##N##i16 index, size_t region_size,             \
      v##N##i16 mask) {                                                        \
    ripple_mask_hvx_gather_i##W##_16((int##W##_t *)dst, (int##W##_t *)src,     \
                                     index, region_size, mask);                \
  }

// Unsigned equivalents: LLVM will have translated these types to (signed) int
#define _decl_unsigned_gather(W, N)                                            \
  RIPPLE_INTRIN_INLINE void ripple_hvx_gather_u##W(                            \
      int##W##_t *dst, const int##W##_t *src, v##N##i32 index,                 \
      size_t region_size) {                                                    \
    ripple_hvx_gather_i##W(dst, src, index, region_size);                      \
  }                                                                            \
  RIPPLE_INTRIN_INLINE void ripple_mask_hvx_gather_u##W(                       \
      int##W##_t *dst, const int##W##_t *src, v##N##i32 index,                 \
      size_t region_size, v##N##i32 mask) {                                    \
    ripple_mask_hvx_gather_i##W(dst, src, index, region_size, mask);           \
  }

#define _decl_unsigned_gather_16(W, N)                                         \
  RIPPLE_INTRIN_INLINE void ripple_hvx_gather_u##W##_16(                       \
      int##W##_t *dst, const int##W##_t *src, v##N##i16 index,                 \
      size_t region_size) {                                                    \
    ripple_hvx_gather_i##W##_16(dst, src, index, region_size);                 \
  }                                                                            \
  RIPPLE_INTRIN_INLINE void ripple_mask_hvx_gather_u##W##_16(                  \
      int##W##_t *dst, const int##W##_t *src, v##N##i16 index,                 \
      size_t region_size, v##N##i16 mask) {                                    \
    ripple_mask_hvx_gather_i##W##_16(dst, src, index, region_size, mask);      \
  }

_decl_float_gather(double, f, 64, 32);
_decl_float_gather(float, f, 32, 32);
_decl_float_gather(_Float16, f, 16, 64);
_decl_float_gather_16(_Float16, f, 16, 64);
#ifndef __has_bf16__
#define __has_bf16__ 0
#endif
#if __has_bf16__
_decl_float_gather(__bf16, bf, 16, 64);
_decl_float_gather_16(__bf16, bf, 16, 64);
#endif

_decl_unsigned_gather(64, 32);
_decl_unsigned_gather(32, 32);
_decl_unsigned_gather(16, 64);
_decl_unsigned_gather(8, 128);
_decl_unsigned_gather_16(16, 64);
_decl_unsigned_gather_16(8, 128)

#undef _decl_float_gather

// ------------------------------ scatter --------------------------------------

/// Declares a typed scatter variant that maps to its int implementation
#define _decl_float_scatter(CT, T, W, N)                                       \
  RIPPLE_INTRIN_INLINE void ripple_hvx_scatter_##T##W(                         \
      CT *dst, v##N##i32 index, v##N##T##W src, size_t region_size) {          \
    v##N##i##W int_src;                                                        \
    __builtin_memcpy(&int_src, &src, sizeof(src));                             \
    ripple_hvx_scatter_i##W((int##W##_t *)dst, index, int_src, region_size);   \
  }                                                                            \
  RIPPLE_INTRIN_INLINE void ripple_mask_hvx_scatter_##T##W(                    \
      CT *dst, v##N##i32 index, v##N##T##W src, size_t region_size,            \
      v##N##i32 mask) {                                                        \
    v##N##i##W int_src;                                                        \
    __builtin_memcpy(&int_src, &src, sizeof(src));                             \
    ripple_mask_hvx_scatter_i##W((int##W##_t *)dst, index, int_src,            \
                                 region_size, mask);                           \
  }

#define _decl_float_scatter_16(CT, T, W, N)                                    \
  RIPPLE_INTRIN_INLINE void ripple_hvx_scatter_##T##W##_16(                    \
      CT *dst, v##N##i16 index, v##N##T##W src, size_t region_size) {          \
    v##N##i##W int_src;                                                        \
    __builtin_memcpy(&int_src, &src, sizeof(src));                             \
    ripple_hvx_scatter_i##W##_16((int##W##_t *)dst, index, int_src,            \
                                 region_size);                                 \
  }                                                                            \
  RIPPLE_INTRIN_INLINE void ripple_mask_hvx_scatter_##T##W##_16(               \
      CT *dst, v##N##i16 index, v##N##T##W src, size_t region_size,            \
      v##N##i16 mask) {                                                        \
    v##N##i##W int_src;                                                        \
    __builtin_memcpy(&int_src, &src, sizeof(src));                             \
    ripple_mask_hvx_scatter_i##W##_16((int##W##_t *)dst, index, int_src,       \
                                      region_size, mask);                      \
  }

// Unsigned equivalents: LLVM will have translated these types to (signed) int
#define _decl_unsigned_scatter(W, N)                                           \
  RIPPLE_INTRIN_INLINE void ripple_hvx_scatter_u##W(                           \
      int##W##_t *dst, v##N##i32 index, v##N##i##W src, size_t region_size) {  \
    ripple_hvx_scatter_i##W(dst, index, src, region_size);                     \
  }                                                                            \
  RIPPLE_INTRIN_INLINE void ripple_mask_hvx_scatter_u##W(                      \
      int##W##_t *dst, v##N##i32 index, v##N##i##W src, size_t region_size,    \
      v##N##i32 mask) {                                                        \
    ripple_mask_hvx_scatter_i##W(dst, index, src, region_size, mask);          \
  }

#define _decl_unsigned_scatter_16(W, N)                                        \
  RIPPLE_INTRIN_INLINE void ripple_hvx_scatter_u##W##_16(                      \
      int##W##_t *dst, v##N##i16 index, v##N##i##W src, size_t region_size) {  \
    ripple_hvx_scatter_i##W##_16(dst, index, src, region_size);                \
  }                                                                            \
  RIPPLE_INTRIN_INLINE void ripple_mask_hvx_scatter_u##W##_16(                 \
      int##W##_t *dst, v##N##i16 index, v##N##i##W src, size_t region_size,    \
      v##N##i16 mask) {                                                        \
    ripple_mask_hvx_scatter_i##W##_16(dst, index, src, region_size, mask);     \
  }

    _decl_float_scatter(double, f, 64, 32);
_decl_float_scatter(float, f, 32, 32);
_decl_float_scatter(_Float16, f, 16, 64);
_decl_float_scatter_16(_Float16, f, 16, 64);
#ifndef __has_bf16__
#define __has_bf16__ 0
#endif
#if __has_bf16__
_decl_float_scatter(__bf16, bf, 16, 64);
_decl_float_scatter_16(__bf16, bf, 16, 64);
#endif

_decl_unsigned_scatter(64, 32);
_decl_unsigned_scatter(32, 32);
_decl_unsigned_scatter(16, 64);
_decl_unsigned_scatter(8, 128);
_decl_unsigned_scatter_16(16, 64);
_decl_unsigned_scatter_16(8, 128)

#undef _decl_float_scatter

} // extern "C"

// _____________________ Local macro name cleanup ______________________________

#undef HVX_GATHER_32
#undef HVX_GATHER_MASKED_32
#undef HVX_GATHER_16
#undef HVX_GATHER_MASKED_16
#undef HVX_SCATTER_32
#undef HVX_SCATTER_MASKED_32
#undef HVX_SCATTER_16
#undef HVX_SCATTER_MASKED_16
