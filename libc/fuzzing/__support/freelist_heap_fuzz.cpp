//===-- freelist_heap_fuzz.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Fuzzing test for llvm-libc freelist-based heap implementation.
///
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/freelist_heap.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include "src/string/memory_utils/inline_memmove.h"
#include "src/string/memory_utils/inline_memset.h"

using LIBC_NAMESPACE::FreeListHeap;
using LIBC_NAMESPACE::inline_memset;
using LIBC_NAMESPACE::cpp::nullopt;
using LIBC_NAMESPACE::cpp::optional;

// Record of an outstanding allocation.
struct Alloc {
  void *ptr;
  size_t size;
  size_t alignment;
  uint8_t canary; // Byte written to the allocation
};

// A simple vector that tracks allocations using the heap.
class AllocVec {
public:
  AllocVec(FreeListHeap &heap) : heap(&heap), size_(0), capacity(0) {
    allocs = nullptr;
  }

  bool empty() const { return !size_; }

  size_t size() const { return size_; }

  bool push_back(Alloc alloc) {
    if (size_ == capacity) {
      size_t new_cap = capacity ? capacity * 2 : 1;
      Alloc *new_allocs = reinterpret_cast<Alloc *>(
          heap->realloc(allocs, new_cap * sizeof(Alloc)));
      if (!new_allocs)
        return false;
      allocs = new_allocs;
      capacity = new_cap;
    }
    allocs[size_++] = alloc;
    return true;
  }

  Alloc &operator[](size_t idx) { return allocs[idx]; }

  void erase_idx(size_t idx) {
    LIBC_NAMESPACE::inline_memmove(&allocs[idx], &allocs[idx + 1],
                                   sizeof(Alloc) * (size_ - idx - 1));
    --size_;
  }

private:
  FreeListHeap *heap;
  Alloc *allocs;
  size_t size_;
  size_t capacity;
};

// Choose a T value by casting libfuzzer data or exit.
template <typename T>
optional<T> choose(const uint8_t *&data, size_t &remainder) {
  if (sizeof(T) > remainder)
    return nullopt;
  T out;
  LIBC_NAMESPACE::inline_memcpy(&out, data, sizeof(T));
  data += sizeof(T);
  remainder -= sizeof(T);
  return out;
}

// The type of allocation to perform
enum class AllocType : uint8_t {
  MALLOC,
  ALIGNED_ALLOC,
  REALLOC,
  CALLOC,
  NUM_ALLOC_TYPES,
};

template <>
optional<AllocType> choose<AllocType>(const uint8_t *&data, size_t &remainder) {
  auto raw = choose<uint8_t>(data, remainder);
  if (!raw)
    return nullopt;
  return static_cast<AllocType>(
      *raw % static_cast<uint8_t>(AllocType::NUM_ALLOC_TYPES));
}

constexpr size_t heap_size = 64 * 1024;

optional<size_t> choose_size(const uint8_t *&data, size_t &remainder) {
  auto raw = choose<size_t>(data, remainder);
  if (!raw)
    return nullopt;
  return *raw % heap_size;
}

optional<size_t> choose_alloc_idx(const AllocVec &allocs, const uint8_t *&data,
                                  size_t &remainder) {
  if (allocs.empty())
    return nullopt;
  auto raw = choose<size_t>(data, remainder);
  if (!raw)
    return nullopt;
  return *raw % allocs.size();
}

#define ASSIGN_OR_RETURN(TYPE, NAME, EXPR)                                     \
  auto maybe_##NAME = EXPR;                                                    \
  if (!maybe_##NAME)                                                           \
    return 0;                                                                  \
  TYPE NAME = *maybe_##NAME

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t remainder) {
  LIBC_NAMESPACE::FreeListHeapBuffer<heap_size> heap;
  AllocVec allocs(heap);

  uint8_t canary = 0;
  while (true) {
    ASSIGN_OR_RETURN(auto, should_alloc, choose<bool>(data, remainder));
    if (should_alloc) {
      ASSIGN_OR_RETURN(auto, alloc_type, choose<AllocType>(data, remainder));
      ASSIGN_OR_RETURN(size_t, alloc_size, choose_size(data, remainder));

      // Perform allocation.
      void *ptr = nullptr;
      size_t alignment = alignof(max_align_t);
      switch (alloc_type) {
      case AllocType::MALLOC:
        ptr = heap.allocate(alloc_size);
        break;
      case AllocType::ALIGNED_ALLOC: {
        ASSIGN_OR_RETURN(size_t, alignment, choose_size(data, remainder));
        alignment = LIBC_NAMESPACE::cpp::bit_ceil(alignment);
        ptr = heap.aligned_allocate(alignment, alloc_size);
        break;
      }
      case AllocType::REALLOC: {
        if (!alloc_size)
          return 0;
        ASSIGN_OR_RETURN(size_t, idx,
                         choose_alloc_idx(allocs, data, remainder));
        Alloc &alloc = allocs[idx];
        ptr = heap.realloc(alloc.ptr, alloc_size);
        if (ptr) {
          // Extend the canary region if necessary.
          if (alloc_size > alloc.size)
            inline_memset(static_cast<char *>(ptr) + alloc.size, alloc.canary,
                          alloc_size - alloc.size);
          alloc.ptr = ptr;
          alloc.size = alloc_size;
          alloc.alignment = alignof(max_align_t);
        }
        break;
      }
      case AllocType::CALLOC: {
        ASSIGN_OR_RETURN(size_t, count, choose_size(data, remainder));
        size_t total;
        if (__builtin_mul_overflow(count, alloc_size, &total))
          return 0;
        ptr = heap.calloc(count, alloc_size);
        if (ptr)
          for (size_t i = 0; i < total; ++i)
            if (static_cast<char *>(ptr)[i] != 0)
              __builtin_trap();
        break;
      }
      case AllocType::NUM_ALLOC_TYPES:
        __builtin_unreachable();
      }

      if (ptr) {
        // aligned_allocate should automatically apply a minimum alignment.
        if (alignment < alignof(max_align_t))
          alignment = alignof(max_align_t);
        // Check alignment.
        if (reinterpret_cast<uintptr_t>(ptr) % alignment)
          __builtin_trap();

        // Reallocation is treated specially above, since we would otherwise
        // lose the original size.
        if (alloc_type != AllocType::REALLOC) {
          // Fill the object with a canary byte.
          inline_memset(ptr, canary, alloc_size);

          // Track the allocation.
          if (!allocs.push_back({ptr, alloc_size, alignment, canary}))
            return 0;
          ++canary;
        }
      }
    } else {
      // Select a random allocation.
      ASSIGN_OR_RETURN(size_t, idx, choose_alloc_idx(allocs, data, remainder));
      Alloc &alloc = allocs[idx];

      // Check alignment.
      if (reinterpret_cast<uintptr_t>(alloc.ptr) % alloc.alignment)
        __builtin_trap();

      // Check the canary.
      uint8_t *ptr = reinterpret_cast<uint8_t *>(alloc.ptr);
      for (size_t i = 0; i < alloc.size; ++i)
        if (ptr[i] != alloc.canary)
          __builtin_trap();

      // Free the allocation and untrack it.
      heap.free(alloc.ptr);
      allocs.erase_idx(idx);
    }
  }
  return 0;
}
