//===-- Utils to test conformance of mem functions ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBC_TEST_SRC_STRING_MEMORY_UTILS_MEMORY_CHECK_UTILS_H
#define LIBC_TEST_SRC_STRING_MEMORY_UTILS_MEMORY_CHECK_UTILS_H

#include "src/__support/CPP/span.h"
#include "src/__support/macros/sanitizer.h"
#include "src/string/memory_utils/utils.h"
#include <assert.h> // assert
#include <stddef.h> // size_t
#include <stdint.h> // uintxx_t
#include <stdlib.h> // malloc/free

namespace __llvm_libc {

// Simple structure to allocate a buffer of a particular size.
// When ASAN is present it also poisons the whole memory.
// This is a utility class to be used by Buffer below, do not use directly.
struct PoisonedBuffer {
  PoisonedBuffer(size_t size) : ptr((char *)malloc(size)) {
    assert(ptr);
    ASAN_POISON_MEMORY_REGION(ptr, size);
  }
  ~PoisonedBuffer() { free(ptr); }

protected:
  char *ptr = nullptr;
};

// Simple structure to allocate a buffer (aligned or not) of a particular size.
// It is backed by a wider buffer that is marked poisoned when ASAN is present.
// The requested region is unpoisoned, this allows catching out of bounds
// accesses.
enum class Aligned : bool { NO = false, YES = true };
struct Buffer : private PoisonedBuffer {
  static constexpr size_t kAlign = 64;
  static constexpr size_t kLeeway = 2 * kAlign;
  Buffer(size_t size, Aligned aligned = Aligned::YES)
      : PoisonedBuffer(size + kLeeway), size(size) {
    offset_ptr = ptr;
    offset_ptr += distance_to_next_aligned<kAlign>(ptr);
    assert((uintptr_t)(offset_ptr) % kAlign == 0);
    if (aligned == Aligned::NO)
      ++offset_ptr;
    assert(offset_ptr > ptr);
    assert((offset_ptr + size) < (ptr + size + kLeeway));
    ASAN_UNPOISON_MEMORY_REGION(offset_ptr, size);
  }
  cpp::span<char> span() { return cpp::span<char>(offset_ptr, size); }

private:
  size_t size = 0;
  char *offset_ptr = nullptr;
};

static inline char GetRandomChar() {
  static constexpr const uint64_t a = 1103515245;
  static constexpr const uint64_t c = 12345;
  static constexpr const uint64_t m = 1ULL << 31;
  static uint64_t seed = 123456789;
  seed = (a * seed + c) % m;
  return static_cast<char>(seed);
}

// Randomize the content of the buffer.
static inline void Randomize(cpp::span<char> buffer) {
  for (auto &current : buffer)
    current = GetRandomChar();
}

// Copy one span to another.
static inline void ReferenceCopy(cpp::span<char> dst,
                                 const cpp::span<char> src) {
  assert(dst.size() == src.size());
  for (size_t i = 0; i < dst.size(); ++i)
    dst[i] = src[i];
}

// Checks that FnImpl implements the memcpy semantic.
template <auto FnImpl>
bool CheckMemcpy(cpp::span<char> dst, cpp::span<char> src, size_t size) {
  assert(dst.size() == src.size());
  assert(dst.size() == size);
  Randomize(dst);
  FnImpl(dst, src, size);
  for (size_t i = 0; i < size; ++i)
    if (dst[i] != src[i])
      return false;
  return true;
}

// Checks that FnImpl implements the memset semantic.
template <auto FnImpl>
bool CheckMemset(cpp::span<char> dst, uint8_t value, size_t size) {
  Randomize(dst);
  FnImpl(dst, value, size);
  for (char c : dst)
    if (c != (char)value)
      return false;
  return true;
}

// Checks that FnImpl implements the bcmp semantic.
template <auto FnImpl>
bool CheckBcmp(cpp::span<char> span1, cpp::span<char> span2, size_t size) {
  assert(span1.size() == span2.size());
  ReferenceCopy(span2, span1);
  // Compare equal
  if (int cmp = FnImpl(span1, span2, size); cmp != 0)
    return false;
  // Compare not equal if any byte differs
  for (size_t i = 0; i < size; ++i) {
    ++span2[i];
    if (int cmp = FnImpl(span1, span2, size); cmp == 0)
      return false;
    if (int cmp = FnImpl(span2, span1, size); cmp == 0)
      return false;
    --span2[i];
  }
  return true;
}

// Checks that FnImpl implements the memcmp semantic.
template <auto FnImpl>
bool CheckMemcmp(cpp::span<char> span1, cpp::span<char> span2, size_t size) {
  assert(span1.size() == span2.size());
  ReferenceCopy(span2, span1);
  // Compare equal
  if (int cmp = FnImpl(span1, span2, size); cmp != 0)
    return false;
  // Compare not equal if any byte differs
  for (size_t i = 0; i < size; ++i) {
    ++span2[i];
    int ground_truth = __builtin_memcmp(span1.data(), span2.data(), size);
    if (ground_truth > 0) {
      if (int cmp = FnImpl(span1, span2, size); cmp <= 0)
        return false;
      if (int cmp = FnImpl(span2, span1, size); cmp >= 0)
        return false;
    } else {
      if (int cmp = FnImpl(span1, span2, size); cmp >= 0)
        return false;
      if (int cmp = FnImpl(span2, span1, size); cmp <= 0)
        return false;
    }
    --span2[i];
  }
  return true;
}

// TODO: Also implement the memmove semantic

} // namespace __llvm_libc

#endif // LIBC_TEST_SRC_STRING_MEMORY_UTILS_MEMORY_CHECK_UTILS_H
