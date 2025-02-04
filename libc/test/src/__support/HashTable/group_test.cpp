//===-- Unittests for control group ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/HashTable/bitmask.h"

#include "src/__support/CPP/bit.h"
#include "src/__support/macros/config.h"
#include "src/stdlib/rand.h"
#include "test/UnitTest/Test.h"
#include <stdint.h>

namespace LIBC_NAMESPACE_DECL {
namespace internal {

struct ByteArray {
  alignas(Group) uint8_t data[sizeof(Group) + 1]{};
};

TEST(LlvmLibcHashTableBitMaskTest, Match) {
  // Any pair of targets have bit differences not only at the lowest bit.
  // No False positive.
  uint8_t targets[4] = {0x00, 0x11, 0xFF, 0x0F};
  size_t count[4] = {0, 0, 0, 0};
  size_t appearance[4][sizeof(Group)];
  ByteArray array{};

  int data[sizeof(uintptr_t) / sizeof(int)];

  for (int &i : data)
    i = rand();

  uintptr_t random = cpp::bit_cast<uintptr_t>(data);

  for (size_t i = 0; i < sizeof(Group); ++i) {
    size_t choice = random % 4;
    random /= 4;
    array.data[i] = targets[choice];
    appearance[choice][count[choice]++] = i;
  }

  for (size_t t = 0; t < sizeof(targets); ++t) {
    auto bitmask = Group::load(array.data).match_byte(targets[t]);
    for (size_t i = 0; i < count[t]; ++i) {
      size_t iterated = 0;
      for (size_t position : bitmask) {
        ASSERT_EQ(appearance[t][iterated], position);
        iterated++;
      }
      ASSERT_EQ(count[t], iterated);
    }
  }
}

TEST(LlvmLibcHashTableBitMaskTest, MaskAvailable) {
  uint8_t values[3] = {0x00, 0x0F, 0x80};

  for (size_t i = 0; i < sizeof(Group); ++i) {
    ByteArray array{};

    int data[sizeof(uintptr_t) / sizeof(int)];

    for (int &j : data)
      j = rand();

    uintptr_t random = cpp::bit_cast<uintptr_t>(data);

    ASSERT_FALSE(Group::load(array.data).mask_available().any_bit_set());

    array.data[i] = 0x80;
    for (size_t j = 0; j < sizeof(Group); ++j) {
      if (i == j)
        continue;
      size_t sample_space = 2 + (j > i);
      size_t choice = random % sample_space;
      random /= sizeof(values);
      array.data[j] = values[choice];
    }

    auto mask = Group::load(array.data).mask_available();
    ASSERT_TRUE(mask.any_bit_set());
    ASSERT_EQ(mask.lowest_set_bit_nonzero(), i);
  }
}
} // namespace internal
} // namespace LIBC_NAMESPACE_DECL
