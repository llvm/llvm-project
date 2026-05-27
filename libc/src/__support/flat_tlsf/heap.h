//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Provide Heap class for the flat_tlsf allocator.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FLAT_TLSF_HEAP_H
#define LLVM_LIBC_SRC___SUPPORT_FLAT_TLSF_HEAP_H

#include "hdr/types/size_t.h"
#include "src/__support/CPP/algorithm.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/flat_tlsf/binning.h"
#include "src/__support/flat_tlsf/bit_utils.h"
#include "src/__support/flat_tlsf/bitfield.h"
#include "src/__support/flat_tlsf/chunk.h"
#include "src/__support/flat_tlsf/common.h"
#include "src/__support/flat_tlsf/node.h"
#include "src/__support/flat_tlsf/tag.h"
#include "src/__support/libc_assert.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/string/memory_utils/inline_memcpy.h"

namespace LIBC_NAMESPACE_DECL {
namespace flat_tlsf {

class Heap {
  BitField available = {};
  Node **gap_list = nullptr;

  struct SearchResult {
    Byte *base;
    Byte *end;
  };

  LIBC_INLINE cpp::optional<SearchResult>
  full_search_bin(uint32_t bin, size_t required_size, size_t align_mask) {
    for (Node *node = gap_list[bin]; node != nullptr; node = node->next) {
      size_t size = chunk::read_word<size_t>(chunk::gap_node_to_size(node));

      Byte *base = chunk::gap_node_to_base(node);
      Byte *end = base + size;
      Byte *aligned_base = bit_utils::align_up_by_mask(base, align_mask);
      if (aligned_base + required_size <= end) {
        deregister_gap(base, size);
        if (base != aligned_base)
          register_gap(base, aligned_base);
        else
          tag::clear_above_free(chunk::end_to_tag(base));
        return SearchResult{aligned_base, end};
      }
    }

    return cpp::nullopt;
  }

  LIBC_INLINE void register_gap(Byte *base, Byte *end) {
    LIBC_ASSERT(chunk::is_chunk_size(base, end));

    size_t size = static_cast<size_t>(end - base);
    uint32_t bin = cpp::min(Binning::size_to_bin(size),
                            static_cast<uint32_t>(Binning::BIN_COUNT - 1));
    Node **bin_ptr = &gap_list[bin];

    if (*bin_ptr == nullptr) {
      LIBC_ASSERT(!available.read_bit(bin));
      available.set_bit(bin);
    }

    chunk::gap_base_to_node(base)->link_at(Node{*bin_ptr, bin_ptr});
    chunk::write_word(chunk::gap_base_to_bin(base), bin);
    chunk::write_word(chunk::gap_base_to_size(base), size);
    chunk::write_word(chunk::gap_end_to_size_and_flag(end), size);

    LIBC_ASSERT(*bin_ptr != nullptr);
  }

  LIBC_INLINE void deregister_gap(Byte *base, size_t size) {
    LIBC_ASSERT(
        gap_list[cpp::min(Binning::size_to_bin(size),
                          static_cast<uint32_t>(Binning::BIN_COUNT - 1))] !=
        nullptr);

    chunk::gap_base_to_node(base)->unlink();

    uint32_t bin = chunk::read_word<uint32_t>(chunk::gap_base_to_bin(base));
    if (gap_list[bin] == nullptr) {
      LIBC_ASSERT(available.read_bit(bin));
      available.clear_bit(bin);
    }
  }

public:
  LIBC_INLINE constexpr Heap() = default;

  // Add an area to be managed by the heap
  LIBC_INLINE Byte *claim(Byte *base, size_t size) {
    Byte *heap_end =
        chunk::align_down(bit_utils::saturating_ptr_add(base, size));
    Byte *heap_base;
    Byte *gap_base;

    // Gap lists haven't been initialized
    if (gap_list == nullptr) {
      base = cpp::max(cpp::bit_cast<Byte *>(uintptr_t{1}), base);
      heap_base = bit_utils::align_up_by(base, alignof(Node *));
      size_t gap_list_size = sizeof(Node *) * Binning::BIN_COUNT;
      gap_base = chunk::align_up(heap_base + gap_list_size + sizeof(Byte));

      // If calculating gap_base overflowed OR the gap_base is higher than
      // heap_end there isn't enough memory to allocate the metadata and cap it
      // off with a tag
      if (gap_base < heap_base || heap_end < gap_base)
        return nullptr;

      Byte tag = tag::ALLOCATED_FLAG;
      if (gap_base < heap_end)
        tag |= tag::ABOVE_FREE_FLAG;
      chunk::write_word(chunk::end_to_tag(gap_base), tag);
      gap_list = reinterpret_cast<Node **>(heap_base);
      for (size_t i = 0; i < Binning::BIN_COUNT; ++i)
        gap_list[i] = nullptr;
    } else {
      // Note that adding the header size and aligning up automatically dodges
      // the possibility of claiming null, if `memory` started at null.
      gap_base = chunk::align_up(base + sizeof(Byte));

      // If calculating gap_base overflowed OR there isn't a CHUNK_UNIT between
      // gap_base and heap_end, then there isn't enough memory to claim
      if (gap_base + CHUNK_UNIT < base || heap_end < gap_base + CHUNK_UNIT)
        return nullptr;

      heap_base = chunk::end_to_tag(gap_base);
      chunk::write_word(heap_base, static_cast<Byte>(tag::ALLOCATED_FLAG |
                                                     tag::ABOVE_FREE_FLAG |
                                                     tag::HEAP_BASE_FLAG));
    }
    if (gap_base < heap_end) {
      register_gap(gap_base, heap_end);
    }

    return heap_end;
  }

  LIBC_INLINE Node *get_gap_list_head(uint32_t bin) const {
    return gap_list[bin];
  }
  LIBC_INLINE Node **get_gap_list_ptr(uint32_t bin) const {
    return &gap_list[bin];
  }
  LIBC_INLINE const BitField &get_available() const { return available; }
  LIBC_INLINE Node **get_gap_list() const { return gap_list; }
  LIBC_INLINE void test_deregister_gap(Byte *base, size_t size) {
    deregister_gap(base, size);
  }

  LIBC_INLINE Byte *allocate(size_t required_size, size_t required_align) {
    size_t required_chunk_size = chunk::required_chunk_size(required_size);
    Byte *base = nullptr;
    Byte *chunk_end = nullptr;
    do {
      size_t bin = Binning::size_to_bin_ceil(
          cpp::max(required_chunk_size, required_align));

      if (bin >= Binning::BIN_COUNT - 1) {
        if (available.read_bit(Binning::BIN_COUNT - 1)) {
          if (auto result =
                  full_search_bin(Binning::BIN_COUNT - 1, required_chunk_size,
                                  required_align - 1)) {
            base = result->base;
            chunk_end = result->end;
            break;
          }
        }
        return nullptr;
      }

      size_t bit = available.bit_scan_after(static_cast<uint32_t>(bin));
      if (bit >= Binning::BIN_COUNT) {
        if (available.read_bit(static_cast<uint32_t>(bin - 1))) {
          if (auto result =
                  full_search_bin(static_cast<uint32_t>(bin - 1),
                                  required_chunk_size, required_align - 1)) {
            base = result->base;
            chunk_end = result->end;
            break;
          }
        }
        return nullptr;
      }

      if (required_align <= CHUNK_UNIT) {
        Node *node_ptr = gap_list[bit];
        size_t size =
            chunk::read_word<size_t>(chunk::gap_node_to_size(node_ptr));

        LIBC_ASSERT(size >= required_chunk_size);
        base = chunk::gap_node_to_base(node_ptr);
        deregister_gap(base, size);
        tag::clear_above_free(chunk::end_to_tag(base));
        chunk_end = base + size;
        break;
      } else {
        size_t align_mask = required_align - 1;
        bool success = false;
        while (true) {
          for (Node *node = gap_list[bit]; node != nullptr; node = node->next) {
            size_t size =
                chunk::read_word<size_t>(chunk::gap_node_to_size(node));
            Byte *b = chunk::gap_node_to_base(node);
            Byte *end = b + size;
            Byte *aligned_base = bit_utils::align_up_by_mask(b, align_mask);
            if (aligned_base + required_chunk_size <= end) {
              deregister_gap(b, size);
              if (b != aligned_base) {
                register_gap(b, aligned_base);
              } else {
                tag::clear_above_free(chunk::end_to_tag(b));
              }
              base = aligned_base;
              chunk_end = end;
              success = true;
              break;
            }
          }
          if (success)
            break;

          if (bit + 1 < Binning::BIN_COUNT ||
              BitField::BITS > Binning::BIN_COUNT) {
            bit = available.bit_scan_after(static_cast<uint32_t>(bit + 1));
            if (bit < Binning::BIN_COUNT)
              continue;
          }

          // Inlined: full_search_bin(bin - 1, required_chunk_size, align_mask)
          for (Node *node = gap_list[bin - 1]; node != nullptr;
               node = node->next) {
            size_t size =
                chunk::read_word<size_t>(chunk::gap_node_to_size(node));
            Byte *b = chunk::gap_node_to_base(node);
            Byte *end = b + size;
            Byte *aligned_base = bit_utils::align_up_by_mask(b, align_mask);
            if (aligned_base + required_chunk_size <= end) {
              deregister_gap(b, size);
              if (b != aligned_base)
                register_gap(b, aligned_base);
              else
                tag::clear_above_free(chunk::end_to_tag(b));

              base = aligned_base;
              chunk_end = end;
              success = true;
              break;
            }
          }
          if (success)
            break;
          return nullptr;
        }
      }
    } while (0);
    LIBC_ASSERT(chunk::align_down(base) == base);

    Byte *end = base + required_chunk_size;
    Byte tag = tag::ALLOCATED_FLAG;
    if (end != chunk_end) {
      register_gap(end, chunk_end);
      tag |= tag::ABOVE_FREE_FLAG;
    }

    chunk::write_word(chunk::end_to_tag(end), tag);
    return base;
  }

  LIBC_INLINE void deallocate(Byte *ptr, size_t required_size) {
    Byte *chunk_base = ptr;
    Byte *chunk_end = chunk::alloc_to_end(chunk_base, required_size);
    Byte tag = chunk::read_word<Byte>(chunk::end_to_tag(chunk_end));

    LIBC_ASSERT(tag::is_allocated(tag));
    LIBC_ASSERT(chunk::is_chunk_size(chunk_base, chunk_end));
    // Try to recombine with a gap below, if it's there.
    // This gap is never the end of the heap, so we don't need to worry about
    // the presence of an end flag.
    Byte *below_tag_ptr = chunk::end_to_tag(chunk_base);
    if (!tag::is_allocated(chunk::read_word<Byte>(below_tag_ptr))) {
      size_t below_size =
          chunk::read_word<size_t>(chunk::gap_end_to_size_and_flag(chunk_base));

      Byte *below_base = chunk_base - below_size;
      deregister_gap(below_base, below_size);
      chunk_base = below_base;
    } else {
      tag::set_above_free(below_tag_ptr);
    }

    // Try to recombine with a gap above, if it's there.
    // The end flag is never clobbered by this operation, so we can still read
    // it later.
    if (tag::is_above_free(tag)) {
      LIBC_ASSERT(!tag::is_heap_end(tag));
      size_t above_size =
          chunk::read_word<size_t>(chunk::gap_base_to_size(chunk_end));
      deregister_gap(chunk_end, above_size);
      chunk_end += above_size;
    }

    register_gap(chunk_base, chunk_end);
  }

  LIBC_INLINE bool try_grow_in_place(Byte *ptr, size_t old_size,
                                     size_t new_size) {
    LIBC_ASSERT(new_size >= old_size);

    Byte *old_end = chunk::alloc_to_end(ptr, old_size);
    Byte *new_end = chunk::alloc_to_end(ptr, new_size);

    if (old_end == new_end)
      return true;

    Byte old_tag = chunk::read_word<Byte>(chunk::end_to_tag(old_end));
    LIBC_ASSERT(tag::is_allocated(old_tag));

    if (tag::is_above_free(old_tag)) {
      size_t above_size =
          chunk::read_word<size_t>(chunk::gap_base_to_size(old_end));
      Byte *above_end = old_end + above_size;

      if (new_end <= above_end) {
        deregister_gap(old_end, above_size);

        if (new_end != above_end) {
          register_gap(new_end, above_end);
          chunk::write_word(
              chunk::end_to_tag(new_end),
              static_cast<Byte>(tag::ALLOCATED_FLAG | tag::ABOVE_FREE_FLAG));
        } else {
          chunk::write_word(chunk::end_to_tag(new_end),
                            static_cast<Byte>(tag::ALLOCATED_FLAG));
        }

        return true;
      }
    }

    return false;
  }

  LIBC_INLINE void shrink_in_place(Byte *ptr, size_t old_size,
                                   size_t new_size) {
    LIBC_ASSERT(new_size != 0);
    LIBC_ASSERT(new_size <= old_size);

    Byte *chunk_end = chunk::alloc_to_end(ptr, old_size);
    Byte *new_end = chunk::alloc_to_end(ptr, new_size);

    if (new_end != chunk_end) {
      Byte old_tag = chunk::read_word<Byte>(chunk::end_to_tag(chunk_end));

      if (tag::is_above_free(old_tag)) {
        size_t above_size =
            chunk::read_word<size_t>(chunk::gap_base_to_size(chunk_end));
        deregister_gap(chunk_end, above_size);
        chunk_end += above_size;
      }

      register_gap(new_end, chunk_end);
      chunk::write_word(
          chunk::end_to_tag(new_end),
          static_cast<Byte>(tag::ALLOCATED_FLAG | tag::ABOVE_FREE_FLAG));
    }
  }

  LIBC_INLINE bool try_reallocate_in_place(Byte *ptr, size_t old_size,
                                           size_t new_size) {
    if (new_size > old_size) {
      return try_grow_in_place(ptr, old_size, new_size);
    } else if (new_size < old_size) {
      shrink_in_place(ptr, old_size, new_size);
      return true;
    } else {
      return true;
    }
  }

  LIBC_INLINE Byte *reallocate(Byte *ptr, size_t old_size, size_t new_size,
                               size_t new_align) {
    if (try_reallocate_in_place(ptr, old_size, new_size))
      return ptr;
    Byte *new_ptr = allocate(new_size, new_align);
    if (new_ptr == nullptr)
      return nullptr;
    inline_memcpy(new_ptr, ptr, old_size);
    deallocate(ptr, old_size);
    return new_ptr;
  }

  static constexpr size_t HEADER_SIZE = sizeof(size_t);

  LIBC_INLINE void *malloc(size_t size) {
    return aligned_alloc(CHUNK_UNIT, size);
  }

  LIBC_INLINE void *aligned_alloc(size_t align, size_t size) {
    if (size == 0)
      return nullptr;

    size_t header_align = alignof(size_t); // 8 bytes
    size_t allocated_align = cpp::max(align, header_align);

    size_t shift = (HEADER_SIZE + align - 1) & ~(align - 1);
    size_t allocated_size = size + shift;

    Byte *base_ptr = allocate(allocated_size, allocated_align);
    if (base_ptr == nullptr)
      return nullptr;

    size_t actual_chunk_size = chunk::required_chunk_size(allocated_size);
    Byte *user_ptr = base_ptr + shift;

    size_t shift_exponent = static_cast<size_t>(cpp::countr_zero(shift));

    size_t *header = reinterpret_cast<size_t *>(user_ptr - HEADER_SIZE);
    *header = actual_chunk_size | (shift_exponent & 31);

    return user_ptr;
  }

  LIBC_INLINE void free(void *ptr) {
    if (ptr == nullptr)
      return;

    Byte *user_ptr = static_cast<Byte *>(ptr);
    size_t *header = reinterpret_cast<size_t *>(user_ptr - HEADER_SIZE);
    size_t header_val = *header;

    size_t shift_exponent = header_val & 31;
    LIBC_ASSERT(shift_exponent >= 3 && "Invalid or double freed pointer");
    if (shift_exponent < 3)
      return;

    size_t shift = size_t{1} << shift_exponent;
    size_t actual_chunk_size = header_val & ~31;

    Byte *base_ptr = user_ptr - shift;

    deallocate(base_ptr, actual_chunk_size - 1);
  }

  LIBC_INLINE void *realloc(void *ptr, size_t new_size) {
    if (ptr == nullptr)
      return malloc(new_size);

    if (new_size == 0) {
      free(ptr);
      return nullptr;
    }

    Byte *user_ptr = static_cast<Byte *>(ptr);
    size_t *header = reinterpret_cast<size_t *>(user_ptr - HEADER_SIZE);
    size_t header_val = *header;

    size_t shift_exponent = header_val & 31;
    if (shift_exponent < 3)
      return nullptr;

    size_t shift = size_t{1} << shift_exponent;
    size_t old_chunk_size = header_val & ~31;

    Byte *base_ptr = user_ptr - shift;

    size_t new_allocated_size = new_size + shift;
    size_t new_chunk_size = chunk::required_chunk_size(new_allocated_size);

    if (try_reallocate_in_place(base_ptr, old_chunk_size - 1,
                                new_chunk_size - 1)) {
      *header = new_chunk_size | shift_exponent;
      return user_ptr;
    }

    Byte *new_user_ptr = static_cast<Byte *>(aligned_alloc(shift, new_size));
    if (new_user_ptr == nullptr)
      return nullptr;

    size_t old_user_size = old_chunk_size - shift - 1;
    size_t bytes_to_copy = cpp::min(old_user_size, new_size);
    inline_memcpy(new_user_ptr, user_ptr, bytes_to_copy);

    free(ptr);
    return new_user_ptr;
  }
};

} // namespace flat_tlsf
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FLAT_TLSF_HEAP_H
