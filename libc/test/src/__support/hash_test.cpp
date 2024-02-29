//===-- Unittests for hash ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/new.h"
#include "src/__support/hash.h"
#include "src/stdlib/rand.h"
#include "src/stdlib/srand.h"
#include "src/string/memset.h"
#include "test/UnitTest/Test.h"

template <class T> struct AlignedMemory {
  T *data;
  size_t offset;
  std::align_val_t alignment;
  AlignedMemory(size_t size, size_t alignment, size_t offset)
      : offset(offset), alignment{alignment} {
    size_t sz = size * sizeof(T);
    size_t aligned = sz + ((-sz) & (alignment - 1)) + alignment;
    LIBC_NAMESPACE::AllocChecker ac;
    data = static_cast<T *>(operator new(aligned, this->alignment, ac));
    data += offset % alignment;
  }
  ~AlignedMemory() { operator delete(data - offset, alignment); }
};

size_t sizes[] = {0, 1, 23, 59, 1024, 5261};
uint8_t values[] = {0, 1, 23, 59, 102, 255};

// Hash value should not change with different alignments.
TEST(LlvmLibcHashTest, SanityCheck) {
  for (size_t sz : sizes) {
    for (uint8_t val : values) {
      uint64_t hash;
      {
        AlignedMemory<char> mem(sz, 64, 0);
        LIBC_NAMESPACE::memset(mem.data, val, sz);
        LIBC_NAMESPACE::internal::HashState state{0x1234567890abcdef};
        state.update(mem.data, sz);
        hash = state.finish();
      }
      for (size_t offset = 1; offset < 64; ++offset) {
        AlignedMemory<char> mem(sz, 64, offset);
        LIBC_NAMESPACE::memset(mem.data, val, sz);
        LIBC_NAMESPACE::internal::HashState state{0x1234567890abcdef};
        state.update(mem.data, sz);
        ASSERT_EQ(hash, state.finish());
      }
    }
  }
}

static inline size_t popcnt(uint64_t x) {
  size_t count = 0;
  while (x) {
    count += x & 1;
    x >>= 1;
  }
  return count;
}

// Mutate a single bit in a rather large input. The hash should change
// significantly. At least one fifth of the bits should not match.
TEST(LlvmLibcHashTest, Avalanche) {
  for (size_t sz : sizes) {
    for (uint8_t val : values) {
      uint64_t hash;
      AlignedMemory<char> mem(sz, 64, 0);
      LIBC_NAMESPACE::memset(mem.data, val, sz);
      {
        LIBC_NAMESPACE::internal::HashState state{0xabcdef1234567890};
        state.update(mem.data, sz);
        hash = state.finish();
      }
      for (size_t i = 0; i < sz; ++i) {
        for (size_t j = 0; j < 8; ++j) {
          uint8_t mask = 1 << j;
          mem.data[i] ^= mask;
          {
            LIBC_NAMESPACE::internal::HashState state{0xabcdef1234567890};
            state.update(mem.data, sz);
            uint64_t new_hash = state.finish();
            ASSERT_GE(popcnt(hash ^ new_hash), size_t{13});
          }
          mem.data[i] ^= mask;
        }
      }
    }
  }
}

// Hash a random sequence of input. The LSB should be uniform enough such that
// values spread across the entire range.
TEST(LlvmLibcHashTest, UniformLSB) {
  LIBC_NAMESPACE::srand(0xffffffff);
  for (size_t sz : sizes) {
    AlignedMemory<size_t> counters(sz, sizeof(size_t), 0);
    LIBC_NAMESPACE::memset(counters.data, 0, sz * sizeof(size_t));
    for (size_t i = 0; i < 200 * sz; ++i) {
      int randomness[8] = {LIBC_NAMESPACE::rand(), LIBC_NAMESPACE::rand(),
                           LIBC_NAMESPACE::rand(), LIBC_NAMESPACE::rand(),
                           LIBC_NAMESPACE::rand(), LIBC_NAMESPACE::rand(),
                           LIBC_NAMESPACE::rand(), LIBC_NAMESPACE::rand()};
      {
        LIBC_NAMESPACE::internal::HashState state{0x1a2b3c4d5e6f7a8b};
        state.update(randomness, sizeof(randomness));
        uint64_t hash = state.finish();
        counters.data[hash % sz]++;
      }
    }
    for (size_t i = 0; i < sz; ++i) {
      ASSERT_GE(counters.data[i], size_t{140});
      ASSERT_LE(counters.data[i], size_t{260});
    }
  }
}

// Hash a low entropy sequence. The MSB should be uniform enough such that
// there is no significant bias even if the value range is small.
// Top 7 bits is examined because it will be used as a secondary key in
// the hash table.
TEST(LlvmLibcHashTest, UniformMSB) {
  size_t sz = 1 << 7;
  AlignedMemory<size_t> counters(sz, sizeof(size_t), 0);
  LIBC_NAMESPACE::memset(counters.data, 0, sz * sizeof(size_t));
  for (size_t i = 0; i < 200 * sz; ++i) {
    LIBC_NAMESPACE::internal::HashState state{0xa1b2c3d4e5f6a7b8};
    state.update(&i, sizeof(i));
    uint64_t hash = state.finish();
    counters.data[hash >> 57]++;
  }
  for (size_t i = 0; i < sz; ++i) {
    ASSERT_GE(counters.data[i], size_t{140});
    ASSERT_LE(counters.data[i], size_t{260});
  }
}
