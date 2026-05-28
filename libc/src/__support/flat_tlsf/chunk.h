//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Provide chunk utility functions for the flat_tlsf allocator.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FLAT_TLSF_CHUNK_H
#define LLVM_LIBC_SRC___SUPPORT_FLAT_TLSF_CHUNK_H

#include "src/__support/endian_internal.h"
#include "src/__support/flat_tlsf/bit_utils.h"
#include "src/__support/flat_tlsf/common.h"
#include "src/__support/flat_tlsf/node.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/string/memory_utils/inline_memcpy.h"

namespace LIBC_NAMESPACE_DECL {
namespace flat_tlsf {
namespace chunk {

LIBC_INLINE bool is_chunk_size(Byte *base, Byte *end) {
  return end >= base + CHUNK_UNIT;
}

LIBC_INLINE constexpr size_t required_chunk_size(size_t size) {
  size_t size_with_tag = size + 1;
  size_t align_offset = (-size_with_tag) & (CHUNK_UNIT - 1);
  return size_with_tag + align_offset;
}

LIBC_INLINE Byte *alloc_to_end(Byte *base, size_t size) {
  return base + required_chunk_size(size);
}

LIBC_INLINE Node *gap_base_to_node(Byte *base) {
  return reinterpret_cast<Node *>(base + GAP_NODE_OFFSET);
}

LIBC_INLINE uint32_t *gap_base_to_bin(Byte *base) {
  return reinterpret_cast<uint32_t *>(base + GAP_BIN_OFFSET);
}

LIBC_INLINE size_t *gap_base_to_size(Byte *base) {
  return reinterpret_cast<size_t *>(base + GAP_LOW_SIZE_OFFSET);
}

// Note: The highest-addressed byte of this size word (at `end - 1`) overlaps
// with the tag byte of the next chunk (retrieved via `end_to_tag(end)`).
// To prevent any tag/flag corruption or endianness issues, the size word is
// always stored and read in Big-Endian format. This guarantees that the LSB
// of the size word is always at the highest address (`end - 1`). Since all
// gap sizes are CHUNK_UNIT aligned, the lower bits of the LSB at `end - 1` are
// naturally 0, keeping the tag/flag status bits safely at 0 (free gap).
LIBC_INLINE size_t *gap_end_to_size_and_flag(Byte *end) {
  return reinterpret_cast<size_t *>(end - GAP_HIGH_SIZE_OFFSET);
}

LIBC_INLINE Byte *gap_node_to_base(Node *node) {
  return reinterpret_cast<Byte *>(node) - GAP_NODE_OFFSET;
}

LIBC_INLINE size_t *gap_node_to_size(Node *node) {
  return reinterpret_cast<size_t *>(reinterpret_cast<Byte *>(node) -
                                    GAP_NODE_OFFSET + GAP_LOW_SIZE_OFFSET);
}

LIBC_INLINE Byte *end_to_tag(Byte *end) { return end - sizeof(Byte); }

LIBC_INLINE Byte *align_up(Byte *ptr) {
  return bit_utils::align_up_by(ptr, CHUNK_UNIT);
}

LIBC_INLINE Byte *align_down(Byte *ptr) {
  return bit_utils::align_down_by(ptr, CHUNK_UNIT);
}

template <class T> LIBC_INLINE T read_word(const void *ptr) {
  T buffer;
  inline_memcpy(&buffer, ptr, sizeof(T));
  return buffer;
}

template <class T> LIBC_INLINE void write_word(void *ptr, T value) {
  inline_memcpy(ptr, &value, sizeof(T));
}

template <class T> LIBC_INLINE T read_big_endian(const void *ptr) {
  return Endian::to_big_endian(read_word<T>(ptr));
}

template <class T> LIBC_INLINE void write_big_endian(void *ptr, T value) {
  write_word<T>(ptr, Endian::to_big_endian(value));
}

} // namespace chunk
} // namespace flat_tlsf
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FLAT_TLSF_CHUNK_H
