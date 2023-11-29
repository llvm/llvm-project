//===-- Fix-sized Monotonic HashTable ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_HASHTABLE_table_H
#define LLVM_LIBC_SRC___SUPPORT_HASHTABLE_table_H

#include "include/llvm-libc-types/ENTRY.h"
#include "src/__support/CPP/new.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/HashTable/bitmask.h"
#include "src/__support/bit.h"
#include "src/__support/hash.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/memory_size.h"
#include "src/string/memset.h"
#include "src/string/strcmp.h"
#include "src/string/strlen.h"
#include <stddef.h>
#include <stdint.h>

namespace LIBC_NAMESPACE {
namespace internal {

LIBC_INLINE uint8_t secondary_hash(uint64_t hash) {
  // top 7 bits of the hash.
  return static_cast<uint8_t>(hash >> 57);
}

// Probe sequence based on triangular numbers, which is guaranteed (since our
// table size is a power of two) to visit every group of elements exactly once.
//
// A triangular probe has us jump by 1 more group every time. So first we
// jump by 1 group (meaning we just continue our linear scan), then 2 groups
// (skipping over 1 group), then 3 groups (skipping over 2 groups), and so on.
//
// If we set sizeof(Group) to be one unit:
//               T[k] = sum {1 + 2 + ... + k} = k * (k + 1) / 2
// It is provable that T[k] mod 2^m generates a permutation of
//                0, 1, 2, 3, ..., 2^m - 2, 2^m - 1
// Detailed proof is available at:
// https://fgiesen.wordpress.com/2015/02/22/triangular-numbers-mod-2n/
struct ProbeSequence {
  size_t position;
  size_t stride;
  size_t entries_mask;

  LIBC_INLINE size_t next() {
    position += stride;
    position &= entries_mask;
    stride += sizeof(Group);
    return position;
  }
};

// The number of entries is at least group width: we do not
// need to do the fixup when we set the control bytes.
// The number of entries is at least 8: we don't have to worry
// about special sizes when check the fullness of the table.
LIBC_INLINE size_t capacity_to_entries(size_t cap) {
  if (8 >= sizeof(Group) && cap < 8)
    return 8;
  if (16 >= sizeof(Group) && cap < 15)
    return 16;
  if (cap < sizeof(Group))
    cap = sizeof(Group);
  // overflow is always checked in allocate()
  return next_power_of_two(cap * 8 / 7);
}

// The heap memory layout for N buckets HashTable is as follows:
//
//             =======================
//             |   N * Entry         |
//             ======================= <- align boundary
//             |   Header            |
//             =======================
//             |   (N + 1) * Byte    |
//             =======================
//
// The trailing group part is to make sure we can always load
// a whole group of control bytes.

struct HashTable {
  HashState state;
  size_t entries_mask;    // number of buckets - 1
  size_t available_slots; // less than capacity
private:
  // How many entries are there in the table.
  LIBC_INLINE size_t num_of_entries() const { return entries_mask + 1; }

  LIBC_INLINE bool is_full() const { return available_slots == 0; }

  LIBC_INLINE size_t offset_from_entries() const {
    size_t entries_size = num_of_entries() * sizeof(ENTRY);
    return entries_size + offset_to(entries_size, table_alignment());
  }

  LIBC_INLINE constexpr static size_t table_alignment() {
    return alignof(HashTable) > alignof(ENTRY) ? alignof(HashTable)
                                               : alignof(ENTRY);
  }

  LIBC_INLINE constexpr static size_t offset_to_groups() {
    return sizeof(HashTable);
  }

  LIBC_INLINE ENTRY &entry(size_t i) {
    return reinterpret_cast<ENTRY *>(this)[-i - 1];
  }

  LIBC_INLINE uint8_t &control(size_t i) {
    uint8_t *ptr = reinterpret_cast<uint8_t *>(this) + offset_to_groups();
    return ptr[i];
  }

  // We duplicate a group of control bytes to the end. Thus, it is possible that
  // we need to set two control bytes at the same time.
  LIBC_INLINE void set_ctrl(size_t index, uint8_t value) {
    size_t index2 = ((index - sizeof(Group)) & entries_mask) + sizeof(Group);
    control(index) = value;
    control(index2) = value;
  }

public:
  LIBC_INLINE static void deallocate(HashTable *table) {
    if (table) {
      void *ptr =
          reinterpret_cast<uint8_t *>(table) - table->offset_from_entries();
      operator delete(ptr, std::align_val_t{table_alignment()});
    }
  }
  LIBC_INLINE static HashTable *allocate(size_t capacity, uint64_t randomness) {
    // check if capacity_to_entries overflows MAX_MEM_SIZE
    if (capacity > size_t{1} << (8 * sizeof(size_t) - 1 - 3))
      return nullptr;
    SafeMemSize entries{capacity_to_entries(capacity)};
    SafeMemSize entries_size = entries * SafeMemSize{sizeof(ENTRY)};
    SafeMemSize align_boundary = entries_size.align_up(table_alignment());
    SafeMemSize ctrl_sizes = entries + SafeMemSize{sizeof(Group)};
    SafeMemSize header_size{offset_to_groups()};
    SafeMemSize total_size =
        (align_boundary + header_size + ctrl_sizes).align_up(table_alignment());
    if (!total_size.valid())
      return nullptr;
    AllocChecker ac;

    void *mem = operator new(total_size, std::align_val_t{table_alignment()},
                             ac);

    HashTable *table = reinterpret_cast<HashTable *>(
        static_cast<uint8_t *>(mem) + align_boundary);
    if (ac) {
      table->entries_mask = entries - 1u;
      table->available_slots = entries / 8 * 7;
      table->state = HashState{randomness};
      memset(&table->control(0), 0x80, ctrl_sizes);
      memset(mem, 0, table->offset_from_entries());
    }
    return table;
  }

private:
  LIBC_INLINE size_t find(const char *key, uint64_t primary) {
    uint8_t secondary = secondary_hash(primary);
    ProbeSequence sequence{static_cast<size_t>(primary), 0, entries_mask};
    while (true) {
      size_t pos = sequence.next();
      Group ctrls = Group::load(&control(pos));
      IteratableBitMask masks = ctrls.match_byte(secondary);
      for (size_t i : masks) {
        size_t index = (pos + i) & entries_mask;
        ENTRY &entry = this->entry(index);
        if (LIBC_LIKELY(entry.key != nullptr && strcmp(entry.key, key) == 0))
          return index;
      }
      BitMask available = ctrls.mask_available();
      // Since there is no deletion, the first time we find an available slot
      // it is also ready to be used as an insertion point. Therefore, we also
      // return the first available slot we find. If such entry is empty, the
      // key will be nullptr.
      if (LIBC_LIKELY(available.any_bit_set())) {
        size_t index =
            (pos + available.lowest_set_bit_nonzero()) & entries_mask;
        return index;
      }
    }
  }

private:
  LIBC_INLINE ENTRY *insert(ENTRY item, uint64_t primary) {
    auto index = find(item.key, primary);
    auto slot = &this->entry(index);
    // SVr4 and POSIX.1-2001 specify that action is significant only for
    // unsuccessful searches, so that an ENTER should not do anything
    // for a successful search.
    if (slot->key != nullptr)
      return slot;

    if (!is_full()) {
      set_ctrl(index, secondary_hash(primary));
      slot->key = item.key;
      slot->data = item.data;
      available_slots--;
      return slot;
    }
    return nullptr;
  }

public:
  LIBC_INLINE ENTRY *find(const char *key) {
    LIBC_NAMESPACE::internal::HashState hasher = state;
    hasher.update(key, strlen(key));
    uint64_t primary = hasher.finish();
    ENTRY &entry = this->entry(find(key, primary));
    if (entry.key == nullptr)
      return nullptr;
    return &entry;
  }
  LIBC_INLINE ENTRY *insert(ENTRY item) {
    LIBC_NAMESPACE::internal::HashState hasher = state;
    hasher.update(item.key, strlen(item.key));
    uint64_t primary = hasher.finish();
    return insert(item, primary);
  }
};
} // namespace internal
} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_HASHTABLE_table_H
