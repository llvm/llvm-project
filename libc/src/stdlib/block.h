//===-- Implementation header for a block of memory -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_BLOCK_H
#define LLVM_LIBC_SRC_STDLIB_BLOCK_H

#include "src/__support/CPP/algorithm.h"
#include "src/__support/CPP/cstddef.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/CPP/new.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/CPP/span.h"
#include "src/__support/CPP/type_traits.h"

#include <stdint.h>

namespace LIBC_NAMESPACE {

namespace internal {
// Types of corrupted blocks, and functions to crash with an error message
// corresponding to each type.
enum class BlockStatus {
  VALID,
  MISALIGNED,
  PREV_MISMATCHED,
  NEXT_MISMATCHED,
};
} // namespace internal

/// Returns the value rounded down to the nearest multiple of alignment.
LIBC_INLINE constexpr size_t align_down(size_t value, size_t alignment) {
  // Note this shouldn't overflow since the result will always be <= value.
  return (value / alignment) * alignment;
}

/// Returns the value rounded down to the nearest multiple of alignment.
template <typename T>
LIBC_INLINE constexpr T *align_down(T *value, size_t alignment) {
  return reinterpret_cast<T *>(
      align_down(reinterpret_cast<size_t>(value), alignment));
}

/// Returns the value rounded up to the nearest multiple of alignment.
LIBC_INLINE constexpr size_t align_up(size_t value, size_t alignment) {
  __builtin_add_overflow(value, alignment - 1, &value);
  return align_down(value, alignment);
}

/// Returns the value rounded up to the nearest multiple of alignment.
template <typename T>
LIBC_INLINE constexpr T *align_up(T *value, size_t alignment) {
  return reinterpret_cast<T *>(
      align_up(reinterpret_cast<size_t>(value), alignment));
}

using ByteSpan = cpp::span<LIBC_NAMESPACE::cpp::byte>;
using cpp::optional;

/// Memory region with links to adjacent blocks.
///
/// The blocks do not encode their size directly. Instead, they encode offsets
/// to the next and previous blocks using the type given by the `OffsetType`
/// template parameter. The encoded offsets are simply the offsets divded by the
/// minimum block alignment, `ALIGNMENT`.
///
/// The `ALIGNMENT` constant provided by the derived block is typically the
/// minimum value of `alignof(OffsetType)`. Since the addressable range of a
/// block is given by `std::numeric_limits<OffsetType>::max() *
/// ALIGNMENT`, it may be advantageous to set a higher alignment if it allows
/// using a smaller offset type, even if this wastes some bytes in order to
/// align block headers.
///
/// Blocks will always be aligned to a `ALIGNMENT` boundary. Block sizes will
/// always be rounded up to a multiple of `ALIGNMENT`.
///
/// As an example, the diagram below represents two contiguous
/// `Block<uint32_t, 8>`s. The indices indicate byte offsets:
///
/// @code{.unparsed}
/// Block 1:
/// +---------------------+------+--------------+
/// | Header              | Info | Usable space |
/// +----------+----------+------+--------------+
/// | prev     | next     |      |              |
/// | 0......3 | 4......7 | 8..9 | 10.......280 |
/// | 00000000 | 00000046 | 8008 |  <app data>  |
/// +----------+----------+------+--------------+
/// Block 2:
/// +---------------------+------+--------------+
/// | Header              | Info | Usable space |
/// +----------+----------+------+--------------+
/// | prev     | next     |      |              |
/// | 0......3 | 4......7 | 8..9 | 10......1056 |
/// | 00000046 | 00000106 | 2008 | f7f7....f7f7 |
/// +----------+----------+------+--------------+
/// @endcode
///
/// The overall size of the block (e.g. 280 bytes) is given by its next offset
/// multiplied by the alignment (e.g. 0x106 * 4). Also, the next offset of a
/// block matches the previous offset of its next block. The first block in a
/// list is denoted by having a previous offset of `0`.
///
/// @tparam   OffsetType  Unsigned integral type used to encode offsets. Larger
///                       types can address more memory, but consume greater
///                       overhead.
/// @tparam   kAlign      Sets the overall alignment for blocks. Minimum is
///                       `alignof(OffsetType)` (the default). Larger values can
///                       address more memory, but consume greater overhead.
template <typename OffsetType = uintptr_t, size_t kAlign = alignof(OffsetType)>
class Block {
public:
  using offset_type = OffsetType;
  static_assert(cpp::is_unsigned_v<offset_type>,
                "offset type must be unsigned");

  static constexpr size_t ALIGNMENT = cpp::max(kAlign, alignof(offset_type));
  static constexpr size_t BLOCK_OVERHEAD = align_up(sizeof(Block), ALIGNMENT);

  // No copy or move.
  Block(const Block &other) = delete;
  Block &operator=(const Block &other) = delete;

  /// Creates the first block for a given memory region.
  static optional<Block *> init(ByteSpan region);

  /// @returns  A pointer to a `Block`, given a pointer to the start of the
  ///           usable space inside the block.
  ///
  /// This is the inverse of `usable_space()`.
  ///
  /// @warning  This method does not do any checking; passing a random
  ///           pointer will return a non-null pointer.
  static Block *from_usable_space(void *usable_space) {
    auto *bytes = reinterpret_cast<cpp::byte *>(usable_space);
    return reinterpret_cast<Block *>(bytes - BLOCK_OVERHEAD);
  }
  static const Block *from_usable_space(const void *usable_space) {
    const auto *bytes = reinterpret_cast<const cpp::byte *>(usable_space);
    return reinterpret_cast<const Block *>(bytes - BLOCK_OVERHEAD);
  }

  /// @returns The total size of the block in bytes, including the header.
  size_t outer_size() const { return next_ * ALIGNMENT; }

  /// @returns The number of usable bytes inside the block.
  size_t inner_size() const { return outer_size() - BLOCK_OVERHEAD; }

  /// @returns The number of bytes requested using AllocFirst or AllocLast.
  size_t requested_size() const { return inner_size() - padding_; }

  /// @returns A pointer to the usable space inside this block.
  cpp::byte *usable_space() {
    return reinterpret_cast<cpp::byte *>(this) + BLOCK_OVERHEAD;
  }
  const cpp::byte *usable_space() const {
    return reinterpret_cast<const cpp::byte *>(this) + BLOCK_OVERHEAD;
  }

  /// Marks the block as free and merges it with any free neighbors.
  ///
  /// This method is static in order to consume and replace the given block
  /// pointer. If neither member is free, the returned pointer will point to the
  /// original block. Otherwise, it will point to the new, larger block created
  /// by merging adjacent free blocks together.
  static void free(Block *&block);

  /// Attempts to split this block.
  ///
  /// If successful, the block will have an inner size of `new_inner_size`,
  /// rounded up to a `ALIGNMENT` boundary. The remaining space will be
  /// returned as a new block.
  ///
  /// This method may fail if the remaining space is too small to hold a new
  /// block. If this method fails for any reason, the original block is
  /// unmodified.
  ///
  /// This method is static in order to consume and replace the given block
  /// pointer with a pointer to the new, smaller block.
  static optional<Block *> split(Block *&block, size_t new_inner_size);

  /// Merges this block with the one that comes after it.
  ///
  /// This method is static in order to consume and replace the given block
  /// pointer with a pointer to the new, larger block.
  static bool merge_next(Block *&block);

  /// Fetches the block immediately after this one.
  ///
  /// For performance, this always returns a block pointer, even if the returned
  /// pointer is invalid. The pointer is valid if and only if `last()` is false.
  ///
  /// Typically, after calling `Init` callers may save a pointer past the end of
  /// the list using `next()`. This makes it easy to subsequently iterate over
  /// the list:
  /// @code{.cpp}
  ///   auto result = Block<>::init(byte_span);
  ///   Block<>* begin = *result;
  ///   Block<>* end = begin->next();
  ///   ...
  ///   for (auto* block = begin; block != end; block = block->next()) {
  ///     // Do something which each block.
  ///   }
  /// @endcode
  Block *next() const;

  /// @copydoc `next`.
  static Block *next_block(const Block *block) {
    return block == nullptr ? nullptr : block->next();
  }

  /// @returns The block immediately before this one, or a null pointer if this
  /// is the first block.
  Block *prev() const;

  /// @copydoc `prev`.
  static Block *prev_block(const Block *block) {
    return block == nullptr ? nullptr : block->prev();
  }

  /// Returns the current alignment of a block.
  size_t alignment() const { return used() ? info_.alignment : 1; }

  /// Indicates whether the block is in use.
  ///
  /// @returns `true` if the block is in use or `false` if not.
  bool used() const { return info_.used; }

  /// Indicates whether this block is the last block or not (i.e. whether
  /// `next()` points to a valid block or not). This is needed because
  /// `next()` points to the end of this block, whether there is a valid
  /// block there or not.
  ///
  /// @returns `true` is this is the last block or `false` if not.
  bool last() const { return info_.last; }

  /// Marks this block as in use.
  void mark_used() { info_.used = 1; }

  /// Marks this block as free.
  void mark_free() { info_.used = 0; }

  /// Marks this block as the last one in the chain.
  constexpr void mark_last() { info_.last = 1; }

  /// Clears the last bit from this block.
  void clear_last() { info_.last = 1; }

  /// @brief Checks if a block is valid.
  ///
  /// @returns `true` if and only if the following conditions are met:
  /// * The block is aligned.
  /// * The prev/next fields match with the previous and next blocks.
  bool is_valid() const {
    return check_status() == internal::BlockStatus::VALID;
  }

  constexpr Block(size_t prev_outer_size, size_t outer_size);

private:
  /// Consumes the block and returns as a span of bytes.
  static ByteSpan as_bytes(Block *&&block);

  /// Consumes the span of bytes and uses it to construct and return a block.
  static Block *as_block(size_t prev_outer_size, ByteSpan bytes);

  /// Returns a `BlockStatus` that is either VALID or indicates the reason why
  /// the block is invalid.
  ///
  /// If the block is invalid at multiple points, this function will only return
  /// one of the reasons.
  internal::BlockStatus check_status() const;

  /// Like `split`, but assumes the caller has already checked to parameters to
  /// ensure the split will succeed.
  static Block *split_impl(Block *&block, size_t new_inner_size);

  /// Offset (in increments of the minimum alignment) from this block to the
  /// previous block. 0 if this is the first block.
  offset_type prev_ = 0;

  /// Offset (in increments of the minimum alignment) from this block to the
  /// next block. Valid even if this is the last block, since it equals the
  /// size of the block.
  offset_type next_ = 0;

  /// Information about the current state of the block:
  /// * If the `used` flag is set, the block's usable memory has been allocated
  ///   and is being used.
  /// * If the `last` flag is set, the block does not have a next block.
  /// * If the `used` flag is set, the alignment represents the requested value
  ///   when the memory was allocated, which may be less strict than the actual
  ///   alignment.
  struct {
    uint16_t used : 1;
    uint16_t last : 1;
    uint16_t alignment : 14;
  } info_;

  /// Number of bytes allocated beyond what was requested. This will be at most
  /// the minimum alignment, i.e. `alignof(offset_type).`
  uint16_t padding_ = 0;
} __attribute__((packed, aligned(kAlign)));

// Public template method implementations.

LIBC_INLINE ByteSpan get_aligned_subspan(ByteSpan bytes, size_t alignment) {
  if (bytes.data() == nullptr)
    return ByteSpan();

  auto unaligned_start = reinterpret_cast<uintptr_t>(bytes.data());
  auto aligned_start = align_up(unaligned_start, alignment);
  auto unaligned_end = unaligned_start + bytes.size();
  auto aligned_end = align_down(unaligned_end, alignment);

  if (aligned_end <= aligned_start)
    return ByteSpan();

  return bytes.subspan(aligned_start - unaligned_start,
                       aligned_end - aligned_start);
}

template <typename OffsetType, size_t kAlign>
optional<Block<OffsetType, kAlign> *>
Block<OffsetType, kAlign>::init(ByteSpan region) {
  optional<ByteSpan> result = get_aligned_subspan(region, ALIGNMENT);
  if (!result)
    return {};

  region = result.value();
  if (region.size() < BLOCK_OVERHEAD)
    return {};

  if (cpp::numeric_limits<OffsetType>::max() < region.size() / ALIGNMENT)
    return {};

  Block *block = as_block(0, region);
  block->mark_last();
  return block;
}

template <typename OffsetType, size_t kAlign>
void Block<OffsetType, kAlign>::free(Block *&block) {
  if (block == nullptr)
    return;

  block->mark_free();
  Block *prev = block->prev();

  if (merge_next(prev))
    block = prev;

  merge_next(block);
}

template <typename OffsetType, size_t kAlign>
optional<Block<OffsetType, kAlign> *>
Block<OffsetType, kAlign>::split(Block *&block, size_t new_inner_size) {
  if (block == nullptr)
    return {};

  if (block->used())
    return {};

  size_t old_inner_size = block->inner_size();
  new_inner_size = align_up(new_inner_size, ALIGNMENT);
  if (old_inner_size < new_inner_size)
    return {};

  if (old_inner_size - new_inner_size < BLOCK_OVERHEAD)
    return {};

  return split_impl(block, new_inner_size);
}

template <typename OffsetType, size_t kAlign>
Block<OffsetType, kAlign> *
Block<OffsetType, kAlign>::split_impl(Block *&block, size_t new_inner_size) {
  size_t prev_outer_size = block->prev_ * ALIGNMENT;
  size_t outer_size1 = new_inner_size + BLOCK_OVERHEAD;
  bool is_last = block->last();
  ByteSpan bytes = as_bytes(cpp::move(block));
  Block *block1 = as_block(prev_outer_size, bytes.subspan(0, outer_size1));
  Block *block2 = as_block(outer_size1, bytes.subspan(outer_size1));

  if (is_last)
    block2->mark_last();
  else
    block2->next()->prev_ = block2->next_;

  block = cpp::move(block1);
  return block2;
}

template <typename OffsetType, size_t kAlign>
bool Block<OffsetType, kAlign>::merge_next(Block *&block) {
  if (block == nullptr)
    return false;

  if (block->last())
    return false;

  Block *next = block->next();
  if (block->used() || next->used())
    return false;

  size_t prev_outer_size = block->prev_ * ALIGNMENT;
  bool is_last = next->last();
  ByteSpan prev_bytes = as_bytes(cpp::move(block));
  ByteSpan next_bytes = as_bytes(cpp::move(next));
  size_t outer_size = prev_bytes.size() + next_bytes.size();
  cpp::byte *merged = ::new (prev_bytes.data()) cpp::byte[outer_size];
  block = as_block(prev_outer_size, ByteSpan(merged, outer_size));

  if (is_last)
    block->mark_last();
  else
    block->next()->prev_ = block->next_;

  return true;
}

template <typename OffsetType, size_t kAlign>
Block<OffsetType, kAlign> *Block<OffsetType, kAlign>::next() const {
  uintptr_t addr =
      last() ? 0 : reinterpret_cast<uintptr_t>(this) + outer_size();
  return reinterpret_cast<Block *>(addr);
}

template <typename OffsetType, size_t kAlign>
Block<OffsetType, kAlign> *Block<OffsetType, kAlign>::prev() const {
  uintptr_t addr =
      (prev_ == 0) ? 0
                   : reinterpret_cast<uintptr_t>(this) - (prev_ * ALIGNMENT);
  return reinterpret_cast<Block *>(addr);
}

// Private template method implementations.

template <typename OffsetType, size_t kAlign>
constexpr Block<OffsetType, kAlign>::Block(size_t prev_outer_size,
                                           size_t outer_size)
    : info_{} {
  prev_ = prev_outer_size / ALIGNMENT;
  next_ = outer_size / ALIGNMENT;
  info_.used = 0;
  info_.last = 0;
  info_.alignment = ALIGNMENT;
}

template <typename OffsetType, size_t kAlign>
ByteSpan Block<OffsetType, kAlign>::as_bytes(Block *&&block) {
  size_t block_size = block->outer_size();
  cpp::byte *bytes = new (cpp::move(block)) cpp::byte[block_size];
  return {bytes, block_size};
}

template <typename OffsetType, size_t kAlign>
Block<OffsetType, kAlign> *
Block<OffsetType, kAlign>::as_block(size_t prev_outer_size, ByteSpan bytes) {
  return ::new (bytes.data()) Block(prev_outer_size, bytes.size());
}

template <typename OffsetType, size_t kAlign>
internal::BlockStatus Block<OffsetType, kAlign>::check_status() const {
  if (reinterpret_cast<uintptr_t>(this) % ALIGNMENT != 0)
    return internal::BlockStatus::MISALIGNED;

  if (!last() && (this >= next() || this != next()->prev()))
    return internal::BlockStatus::NEXT_MISMATCHED;

  if (prev() && (this <= prev() || this != prev()->next()))
    return internal::BlockStatus::PREV_MISMATCHED;

  return internal::BlockStatus::VALID;
}

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_STDLIB_BLOCK_H
