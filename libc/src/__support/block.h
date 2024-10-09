//===-- Implementation header for a block of memory -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_BLOCK_H
#define LLVM_LIBC_SRC___SUPPORT_BLOCK_H

#include "src/__support/CPP/algorithm.h"
#include "src/__support/CPP/cstddef.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/CPP/new.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/CPP/span.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/libc_assert.h"
#include "src/__support/macros/config.h"

#include <stdint.h>

namespace LIBC_NAMESPACE_DECL {

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
/// The blocks store their offsets to the previous and next blocks. The latter
/// is also the block's size.
///
/// The `ALIGNMENT` constant provided by the derived block is typically the
/// minimum value of `alignof(OffsetType)`. Blocks will always be aligned to a
/// `ALIGNMENT` boundary. Block sizes will always be rounded up to a multiple of
/// `ALIGNMENT`.
///
/// As an example, the diagram below represents two contiguous
/// `Block<uint32_t, 8>`s. The indices indicate byte offsets:
///
/// @code{.unparsed}
/// Block 1:
/// +---------------------+--------------+
/// | Header              | Usable space |
/// +----------+----------+--------------+
/// | prev     | next     |              |
/// | 0......3 | 4......7 | 8........227 |
/// | 00000000 | 00000230 |  <app data>  |
/// +----------+----------+--------------+
/// Block 2:
/// +---------------------+--------------+
/// | Header              | Usable space |
/// +----------+----------+--------------+
/// | prev     | next     |              |
/// | 0......3 | 4......7 | 8........827 |
/// | 00000230 | 00000830 | f7f7....f7f7 |
/// +----------+----------+--------------+
/// @endcode
///
/// As a space optimization, when a block is allocated, it consumes the prev
/// field of the following block:
///
/// Block 1 (used):
/// +---------------------+--------------+
/// | Header              | Usable space |
/// +----------+----------+--------------+
/// | prev     | next     |              |
/// | 0......3 | 4......7 | 8........230 |
/// | 00000000 | 00000230 |  <app data>  |
/// +----------+----------+--------------+
/// Block 2:
/// +---------------------+--------------+
/// | B1       | Header   | Usable space |
/// +----------+----------+--------------+
/// |          | next     |              |
/// | 0......3 | 4......7 | 8........827 |
/// | xxxxxxxx | 00000830 | f7f7....f7f7 |
/// +----------+----------+--------------+
///
/// The next offset of a block matches the previous offset of its next block.
/// The first block in a list is denoted by having a previous offset of `0`.
///
/// @tparam   OffsetType  Unsigned integral type used to encode offsets. Larger
///                       types can address more memory, but consume greater
///                       overhead.
/// @tparam   kAlign      Sets the overall alignment for blocks. Minimum is
///                       `alignof(OffsetType)`, but the default is max_align_t,
///                       since the usable space will then already be
///                       aligned to max_align_t if the size of OffsetType is no
///                       less than half of max_align_t. Larger values cause
///                       greater overhead.
template <typename OffsetType = uintptr_t, size_t kAlign = alignof(max_align_t)>
class Block {
  // Masks for the contents of the next_ field.
  static constexpr size_t PREV_FREE_MASK = 1 << 0;
  static constexpr size_t LAST_MASK = 1 << 1;
  static constexpr size_t SIZE_MASK = ~(PREV_FREE_MASK | LAST_MASK);

public:
  using offset_type = OffsetType;
  static_assert(cpp::is_unsigned_v<offset_type>,
                "offset type must be unsigned");
  static constexpr size_t ALIGNMENT =
      cpp::max(cpp::max(kAlign, alignof(offset_type)), size_t{4});
  static constexpr size_t BLOCK_OVERHEAD = align_up(sizeof(Block), ALIGNMENT);

  // No copy or move.
  Block(const Block &other) = delete;
  Block &operator=(const Block &other) = delete;

  /// Creates the first block for a given memory region, followed by a sentinel
  /// last block. Returns the first block.
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
  size_t outer_size() const { return next_ & SIZE_MASK; }

  static size_t outer_size(size_t inner_size) {
    // The usable region includes the prev_ field of the next block.
    return inner_size - sizeof(prev_) + BLOCK_OVERHEAD;
  }

  /// @returns The number of usable bytes inside the block.
  size_t inner_size() const {
    if (!next())
      return 0;
    return inner_size(outer_size());
  }

  static size_t inner_size(size_t outer_size) {
    // The usable region includes the prev_ field of the next block.
    return outer_size - BLOCK_OVERHEAD + sizeof(prev_);
  }

  /// @returns A pointer to the usable space inside this block.
  cpp::byte *usable_space() {
    return reinterpret_cast<cpp::byte *>(this) + BLOCK_OVERHEAD;
  }
  const cpp::byte *usable_space() const {
    return reinterpret_cast<const cpp::byte *>(this) + BLOCK_OVERHEAD;
  }

  // @returns The region of memory the block manages, including the header.
  ByteSpan region() {
    return {reinterpret_cast<cpp::byte *>(this), outer_size()};
  }

  /// Attempts to split this block.
  ///
  /// If successful, the block will have an inner size of `new_inner_size`,
  /// rounded to ensure that the split point is on an ALIGNMENT boundary. The
  /// remaining space will be returned as a new block. Note that the prev_ field
  /// of the next block counts as part of the inner size of the returnd block.
  ///
  /// This method may fail if the remaining space is too small to hold a new
  /// block. If this method fails for any reason, the original block is
  /// unmodified.
  optional<Block *> split(size_t new_inner_size);

  /// Merges this block with the one that comes after it.
  bool merge_next();

  /// @returns The block immediately after this one, or a null pointer if this
  /// is the last block.
  Block *next() const;

  /// @returns The free block immediately before this one, otherwise nullptr.
  Block *prev_free() const;

  /// @returns Whether the block is unavailable for allocation.
  bool used() const { return !next() || !next()->prev_free(); }

  /// Marks this block as in use.
  void mark_used() {
    LIBC_ASSERT(next() && "last block is always considered used");
    next()->next_ &= ~PREV_FREE_MASK;
  }

  /// Marks this block as free.
  void mark_free() {
    LIBC_ASSERT(next() && "last block is always considered used");
    next()->next_ |= PREV_FREE_MASK;
    // The next block's prev_ field becomes alive, as it is no longer part of
    // this block's used space.
    *new (&next()->prev_) offset_type = outer_size();
  }

  /// Marks this block as the last one in the chain. Makes next() return
  /// nullptr.
  void mark_last() { next_ |= LAST_MASK; }

  constexpr Block(size_t outer_size);

  bool is_usable_space_aligned(size_t alignment) const {
    return reinterpret_cast<uintptr_t>(usable_space()) % alignment == 0;
  }

  /// @returns The new inner size of this block that would give the usable
  /// space of the next block the given alignment.
  size_t padding_for_alignment(size_t alignment) const {
    if (is_usable_space_aligned(alignment))
      return 0;

    // We need to ensure we can always split this block into a "padding" block
    // and the aligned block. To do this, we need enough extra space for at
    // least one block.
    //
    // |block   |usable_space                          |
    // |........|......................................|
    //                            ^
    //                            Alignment requirement
    //
    //
    // |block   |space   |block   |usable_space        |
    // |........|........|........|....................|
    //                            ^
    //                            Alignment requirement
    //
    alignment = cpp::max(alignment, ALIGNMENT);
    uintptr_t start = reinterpret_cast<uintptr_t>(usable_space());
    uintptr_t next_usable_space = align_up(start + BLOCK_OVERHEAD, alignment);
    uintptr_t next_block = next_usable_space - BLOCK_OVERHEAD;
    return next_block - start + sizeof(prev_);
  }

  // Check that we can `allocate` a block with a given alignment and size from
  // this existing block.
  bool can_allocate(size_t alignment, size_t size) const;

  // This is the return type for `allocate` which can split one block into up to
  // three blocks.
  struct BlockInfo {
    // This is the newly aligned block. It will have the alignment requested by
    // a call to `allocate` and at most `size`.
    Block *block;

    // If the usable_space in the new block was not aligned according to the
    // `alignment` parameter, we will need to split into this block and the
    // `block` to ensure `block` is properly aligned. In this case, `prev` will
    // be a pointer to this new "padding" block. `prev` will be nullptr if no
    // new block was created or we were able to merge the block before the
    // original block with the "padding" block.
    Block *prev;

    // This is the remainder of the next block after splitting the `block`
    // according to `size`. This can happen if there's enough space after the
    // `block`.
    Block *next;
  };

  // Divide a block into up to 3 blocks according to `BlockInfo`. This should
  // only be called if `can_allocate` returns true.
  static BlockInfo allocate(Block *block, size_t alignment, size_t size);

private:
  /// Construct a block to represent a span of bytes. Overwrites only enough
  /// memory for the block header; the rest of the span is left alone.
  static Block *as_block(ByteSpan bytes);

  /// Like `split`, but assumes the caller has already checked to parameters to
  /// ensure the split will succeed.
  Block *split_impl(size_t new_inner_size);

  /// Offset from this block to the previous block. 0 if this is the first
  /// block. This field is only alive when the previous block is free;
  /// otherwise, its memory is reused as part of the previous block's usable
  /// space.
  offset_type prev_ = 0;

  /// Offset from this block to the next block. Valid even if this is the last
  /// block, since it equals the size of the block.
  offset_type next_ = 0;

  /// Information about the current state of the block is stored in the two low
  /// order bits of the next_ value. These are guaranteed free by a minimum
  /// alignment (and thus, alignment of the size) of 4. The lowest bit is the
  /// `prev_free` flag, and the other bit is the `last` flag.
  ///
  /// * If the `prev_free` flag is set, the block isn't the first and the
  ///   previous block is free.
  /// * If the `last` flag is set, the block is the sentinel last block. It is
  ///   summarily considered used and has no next block.
} __attribute__((packed, aligned(cpp::max(kAlign, size_t{4}))));

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
  // Two blocks are allocated: a free block and a sentinel last block.
  if (region.size() < 2 * BLOCK_OVERHEAD)
    return {};

  if (cpp::numeric_limits<OffsetType>::max() < region.size())
    return {};

  Block *block = as_block(region.first(region.size() - BLOCK_OVERHEAD));
  Block *last = as_block(region.last(BLOCK_OVERHEAD));
  block->mark_free();
  last->mark_last();
  return block;
}

template <typename OffsetType, size_t kAlign>
bool Block<OffsetType, kAlign>::can_allocate(size_t alignment,
                                             size_t size) const {
  if (inner_size() < size)
    return false;
  if (is_usable_space_aligned(alignment))
    return true;

  // Alignment isn't met, so a padding block is needed. Determine amount of
  // inner_size() consumed by the padding block.
  size_t padding_size = padding_for_alignment(alignment) - sizeof(prev_);

  // Check that there is room for the allocation in the following aligned block.
  size_t aligned_inner_size = inner_size() - padding_size - BLOCK_OVERHEAD;
  return size <= aligned_inner_size;
}

template <typename OffsetType, size_t kAlign>
typename Block<OffsetType, kAlign>::BlockInfo
Block<OffsetType, kAlign>::allocate(Block *block, size_t alignment,
                                    size_t size) {
  LIBC_ASSERT(
      block->can_allocate(alignment, size) &&
      "Calls to this function for a given alignment and size should only be "
      "done if `can_allocate` for these parameters returns true.");

  BlockInfo info{block, /*prev=*/nullptr, /*next=*/nullptr};

  if (!info.block->is_usable_space_aligned(alignment)) {
    Block *original = info.block;
    optional<Block *> maybe_aligned_block =
        original->split(info.block->padding_for_alignment(alignment));
    LIBC_ASSERT(maybe_aligned_block.has_value() &&
                "This split should always result in a new block. The check in "
                "`can_allocate` ensures that we have enough space here to make "
                "two blocks.");

    if (Block *prev = original->prev_free()) {
      // If there is a free block before this, we can merge the current one with
      // the newly created one.
      prev->merge_next();
    } else {
      info.prev = original;
    }

    Block *aligned_block = *maybe_aligned_block;
    LIBC_ASSERT(aligned_block->is_usable_space_aligned(alignment) &&
                "The aligned block isn't aligned somehow.");
    info.block = aligned_block;
  }

  // Now get a block for the requested size.
  if (optional<Block *> next = info.block->split(size))
    info.next = *next;

  return info;
}

template <typename OffsetType, size_t kAlign>
optional<Block<OffsetType, kAlign> *>
Block<OffsetType, kAlign>::split(size_t new_inner_size) {
  if (used())
    return {};
  // The prev_ field of the next block is always available, so there is a
  // minimum size to a block created through splitting.
  if (new_inner_size < sizeof(prev_))
    return {};

  size_t old_inner_size = inner_size();
  new_inner_size =
      align_up(new_inner_size - sizeof(prev_), ALIGNMENT) + sizeof(prev_);
  if (old_inner_size < new_inner_size)
    return {};

  if (old_inner_size - new_inner_size < BLOCK_OVERHEAD)
    return {};

  return split_impl(new_inner_size);
}

template <typename OffsetType, size_t kAlign>
Block<OffsetType, kAlign> *
Block<OffsetType, kAlign>::split_impl(size_t new_inner_size) {
  size_t outer_size1 = outer_size(new_inner_size);
  LIBC_ASSERT(outer_size1 % ALIGNMENT == 0 && "new size must be aligned");
  ByteSpan new_region = region().subspan(outer_size1);
  next_ &= ~SIZE_MASK;
  next_ |= outer_size1;

  Block *new_block = as_block(new_region);
  mark_free(); // Free status for this block is now stored in new_block.
  new_block->next()->prev_ = new_region.size();
  return new_block;
}

template <typename OffsetType, size_t kAlign>
bool Block<OffsetType, kAlign>::merge_next() {
  if (used() || next()->used())
    return false;
  size_t new_size = outer_size() + next()->outer_size();
  next_ &= ~SIZE_MASK;
  next_ |= new_size;
  next()->prev_ = new_size;
  return true;
}

template <typename OffsetType, size_t kAlign>
Block<OffsetType, kAlign> *Block<OffsetType, kAlign>::next() const {
  if (next_ & LAST_MASK)
    return nullptr;
  return reinterpret_cast<Block *>(reinterpret_cast<uintptr_t>(this) +
                                   outer_size());
}

template <typename OffsetType, size_t kAlign>
Block<OffsetType, kAlign> *Block<OffsetType, kAlign>::prev_free() const {
  if (!(next_ & PREV_FREE_MASK))
    return nullptr;
  return reinterpret_cast<Block *>(reinterpret_cast<uintptr_t>(this) - prev_);
}

// Private template method implementations.

template <typename OffsetType, size_t kAlign>
constexpr Block<OffsetType, kAlign>::Block(size_t outer_size)
    : next_(outer_size) {
  LIBC_ASSERT(outer_size % ALIGNMENT == 0 && "block sizes must be aligned");
}

template <typename OffsetType, size_t kAlign>
Block<OffsetType, kAlign> *Block<OffsetType, kAlign>::as_block(ByteSpan bytes) {
  return ::new (bytes.data()) Block(bytes.size());
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_BLOCK_H
