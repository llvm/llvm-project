//===-- asan_oob_test.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
//===----------------------------------------------------------------------===//
#include "asan_test_utils.h"

NOINLINE void asan_write_sized_aligned(uint8_t *p, size_t size) {
  EXPECT_EQ(0U, ((uintptr_t)p % size));
  if      (size == 1) asan_write((uint8_t*)p);
  else if (size == 2) asan_write((uint16_t*)p);
  else if (size == 4) asan_write((uint32_t*)p);
  else if (size == 8) asan_write((uint64_t*)p);
}

template<typename T>
NOINLINE void oob_test(int size, int off) {
  char *p = (char*)malloc_aaa(size);
  // fprintf(stderr, "writing %d byte(s) into [%p,%p) with offset %d\n",
  //        sizeof(T), p, p + size, off);
  asan_write((T*)(p + off));
  free_aaa(p);
}

template<typename T>
void OOBTest() {
  char expected_str[100];
  for (int size = sizeof(T); size < 20; size += 5) {
    for (int i = -5; i < 0; i++) {
      const char *str =
          "is located.*%d byte.*to the left";
      sprintf(expected_str, str, abs(i));
      EXPECT_DEATH(oob_test<T>(size, i), expected_str);
    }

    for (int i = 0; i < (int)(size - sizeof(T) + 1); i++)
      oob_test<T>(size, i);

    for (int i = size - sizeof(T) + 1; i <= (int)(size + 2 * sizeof(T)); i++) {
      const char *str =
          "is located.*%d byte.*to the right";
      int off = i >= size ? (i - size) : 0;
      // we don't catch unaligned partially OOB accesses.
      if (i % sizeof(T)) continue;
      sprintf(expected_str, str, off);
      EXPECT_DEATH(oob_test<T>(size, i), expected_str);
    }
  }

  EXPECT_DEATH(oob_test<T>(kLargeMalloc, -1),
          "is located.*1 byte.*to the left");
  EXPECT_DEATH(oob_test<T>(kLargeMalloc, kLargeMalloc),
          "is located.*0 byte.*to the right");
}

// TODO(glider): the following tests are EXTREMELY slow on Darwin:
//   AddressSanitizer.OOB_char (125503 ms)
//   AddressSanitizer.OOB_int (126890 ms)
//   AddressSanitizer.OOBRightTest (315605 ms)
//   AddressSanitizer.SimpleStackTest (366559 ms)

TEST(AddressSanitizer, OOB_char) {
  OOBTest<U1>();
}

TEST(AddressSanitizer, OOB_int) {
  OOBTest<U4>();
}

TEST(AddressSanitizer, OOBRightTest) {
  size_t max_access_size = SANITIZER_WORDSIZE == 64 ? 8 : 4;
  for (size_t access_size = 1; access_size <= max_access_size;
       access_size *= 2) {
    for (size_t alloc_size = 1; alloc_size <= 8; alloc_size++) {
      for (size_t offset = 0; offset <= 8; offset += access_size) {
        void *p = malloc(alloc_size);
        // allocated: [p, p + alloc_size)
        // accessed:  [p + offset, p + offset + access_size)
        uint8_t *addr = (uint8_t*)p + offset;
        if (offset + access_size <= alloc_size) {
          asan_write_sized_aligned(addr, access_size);
        } else {
          int outside_bytes = offset > alloc_size ? (offset - alloc_size) : 0;
          const char *str =
              "is located.%d *byte.*to the right";
          char expected_str[100];
          sprintf(expected_str, str, outside_bytes);
          EXPECT_DEATH(asan_write_sized_aligned(addr, access_size),
                       expected_str);
        }
        free(p);
      }
    }
  }
}

TEST(AddressSanitizer, LargeOOBRightTest) {
  size_t large_power_of_two = 1 << 19;
  for (size_t i = 16; i <= 256; i *= 2) {
    size_t size = large_power_of_two - i;
    char *p = Ident(new char[size]);
    EXPECT_DEATH(p[size] = 0, "is located 0 bytes to the right");
    delete [] p;
  }
}

TEST(AddressSanitizer, DISABLED_DemoOOBLeftLow) {
  oob_test<U1>(10, -1);
}

TEST(AddressSanitizer, DISABLED_DemoOOBLeftHigh) {
  oob_test<U1>(kLargeMalloc, -1);
}

TEST(AddressSanitizer, DISABLED_DemoOOBRightLow) {
  oob_test<U1>(10, 10);
}

TEST(AddressSanitizer, DISABLED_DemoOOBRightHigh) {
  oob_test<U1>(kLargeMalloc, kLargeMalloc);
}
