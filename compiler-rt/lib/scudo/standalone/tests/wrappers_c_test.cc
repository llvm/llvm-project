//===-- wrappers_c_test.cc --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "platform.h"

#include "gtest/gtest.h"

#include <limits.h>
#include <malloc.h>
#include <unistd.h>

// Note that every C allocation function in the test binary will be fulfilled
// by Scudo (this includes the gtest APIs, etc.), which is a test by itself.
// But this might also lead to unexpected side-effects, since the allocation and
// deallocation operations in the TEST functions will coexist with others (see
// the EXPECT_DEATH comment below).

// We have to use a small quarantine to make sure that our double-free tests
// trigger. Otherwise EXPECT_DEATH ends up reallocating the chunk that was just
// freed (this depends on the size obviously) and the following free succeeds.
extern "C" __attribute__((visibility("default"))) const char *
__scudo_default_options() {
  return "quarantine_size_kb=256:thread_local_quarantine_size_kb=128:"
         "quarantine_max_chunk_size=512";
}

static const size_t Size = 100U;

TEST(ScudoWrappersCTest, Malloc) {
  void *P = malloc(Size);
  EXPECT_NE(P, nullptr);
  EXPECT_LE(Size, malloc_usable_size(P));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(P) % FIRST_32_SECOND_64(8U, 16U), 0U);
  EXPECT_DEATH(
      free(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(P) | 1U)), "");
  free(P);
  EXPECT_DEATH(free(P), "");

  P = malloc(0U);
  EXPECT_NE(P, nullptr);
  free(P);

  errno = 0;
  EXPECT_EQ(malloc(SIZE_MAX), nullptr);
  EXPECT_EQ(errno, ENOMEM);
}

TEST(ScudoWrappersCTest, Calloc) {
  void *P = calloc(1U, Size);
  EXPECT_NE(P, nullptr);
  EXPECT_LE(Size, malloc_usable_size(P));
  for (size_t I = 0; I < Size; I++)
    EXPECT_EQ((reinterpret_cast<uint8_t *>(P))[I], 0U);
  free(P);

  P = calloc(1U, 0U);
  EXPECT_NE(P, nullptr);
  free(P);
  P = calloc(0U, 1U);
  EXPECT_NE(P, nullptr);
  free(P);

  errno = 0;
  EXPECT_EQ(calloc(SIZE_MAX, 1U), nullptr);
  EXPECT_EQ(errno, ENOMEM);
  errno = 0;
  EXPECT_EQ(calloc(static_cast<size_t>(LONG_MAX) + 1U, 2U), nullptr);
  if (SCUDO_ANDROID)
    EXPECT_EQ(errno, ENOMEM);
  errno = 0;
  EXPECT_EQ(calloc(SIZE_MAX, SIZE_MAX), nullptr);
  EXPECT_EQ(errno, ENOMEM);
}

TEST(ScudoWrappersCTest, Memalign) {
  void *P;
  for (size_t I = FIRST_32_SECOND_64(2U, 3U); I <= 18U; I++) {
    const size_t Alignment = 1U << I;

    P = memalign(Alignment, Size);
    EXPECT_NE(P, nullptr);
    EXPECT_LE(Size, malloc_usable_size(P));
    EXPECT_EQ(reinterpret_cast<uintptr_t>(P) % Alignment, 0U);
    free(P);

    P = nullptr;
    EXPECT_EQ(posix_memalign(&P, Alignment, Size), 0);
    EXPECT_NE(P, nullptr);
    EXPECT_LE(Size, malloc_usable_size(P));
    EXPECT_EQ(reinterpret_cast<uintptr_t>(P) % Alignment, 0U);
    free(P);
  }

  EXPECT_EQ(memalign(4096U, SIZE_MAX), nullptr);
  EXPECT_EQ(posix_memalign(&P, 15U, Size), EINVAL);
  EXPECT_EQ(posix_memalign(&P, 4096U, SIZE_MAX), ENOMEM);

  // Android's memalign accepts non power-of-2 alignments, and 0.
  if (SCUDO_ANDROID) {
    for (size_t Alignment = 0U; Alignment <= 128U; Alignment++) {
      P = memalign(Alignment, 1024U);
      EXPECT_NE(P, nullptr);
      free(P);
    }
  }
}

TEST(ScudoWrappersCTest, AlignedAlloc) {
  const size_t Alignment = 4096U;
  void *P = aligned_alloc(Alignment, Alignment * 4U);
  EXPECT_NE(P, nullptr);
  EXPECT_LE(Alignment * 4U, malloc_usable_size(P));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(P) % Alignment, 0U);
  free(P);

  errno = 0;
  P = aligned_alloc(Alignment, Size);
  EXPECT_EQ(P, nullptr);
  EXPECT_EQ(errno, EINVAL);
}

TEST(ScudoWrappersCTest, Realloc) {
  // realloc(nullptr, N) is malloc(N)
  void *P = realloc(nullptr, 0U);
  EXPECT_NE(P, nullptr);
  free(P);

  P = malloc(Size);
  EXPECT_NE(P, nullptr);
  // realloc(P, 0U) is free(P) and returns nullptr
  EXPECT_EQ(realloc(P, 0U), nullptr);

  P = malloc(Size);
  EXPECT_NE(P, nullptr);
  EXPECT_LE(Size, malloc_usable_size(P));
  memset(P, 0x42, Size);

  P = realloc(P, Size * 2U);
  EXPECT_NE(P, nullptr);
  EXPECT_LE(Size * 2U, malloc_usable_size(P));
  for (size_t I = 0; I < Size; I++)
    EXPECT_EQ(0x42, (reinterpret_cast<uint8_t *>(P))[I]);

  P = realloc(P, Size / 2U);
  EXPECT_NE(P, nullptr);
  EXPECT_LE(Size / 2U, malloc_usable_size(P));
  for (size_t I = 0; I < Size / 2U; I++)
    EXPECT_EQ(0x42, (reinterpret_cast<uint8_t *>(P))[I]);
  free(P);

  EXPECT_DEATH(P = realloc(P, Size), "");

  errno = 0;
  EXPECT_EQ(realloc(nullptr, SIZE_MAX), nullptr);
  EXPECT_EQ(errno, ENOMEM);
  P = malloc(Size);
  EXPECT_NE(P, nullptr);
  errno = 0;
  EXPECT_EQ(realloc(P, SIZE_MAX), nullptr);
  EXPECT_EQ(errno, ENOMEM);
  free(P);

  // Android allows realloc of memalign pointers.
  if (SCUDO_ANDROID) {
    const size_t Alignment = 1024U;
    P = memalign(Alignment, Size);
    EXPECT_NE(P, nullptr);
    EXPECT_LE(Size, malloc_usable_size(P));
    EXPECT_EQ(reinterpret_cast<uintptr_t>(P) % Alignment, 0U);
    memset(P, 0x42, Size);

    P = realloc(P, Size * 2U);
    EXPECT_NE(P, nullptr);
    EXPECT_LE(Size * 2U, malloc_usable_size(P));
    for (size_t I = 0; I < Size; I++)
      EXPECT_EQ(0x42, (reinterpret_cast<uint8_t *>(P))[I]);
    free(P);
  }
}

#ifndef M_DECAY_TIME
#define M_DECAY_TIME -100
#endif

#ifndef M_PURGE
#define M_PURGE -101
#endif

TEST(ScudoWrappersCTest, Mallopt) {
  errno = 0;
  EXPECT_EQ(mallopt(-1000, 1), 0);
  // mallopt doesn't set errno.
  EXPECT_EQ(errno, 0);

  EXPECT_EQ(mallopt(M_PURGE, 0), 1);

  EXPECT_EQ(mallopt(M_DECAY_TIME, 1), 1);
  EXPECT_EQ(mallopt(M_DECAY_TIME, 0), 1);
  EXPECT_EQ(mallopt(M_DECAY_TIME, 1), 1);
  EXPECT_EQ(mallopt(M_DECAY_TIME, 0), 1);
}

TEST(ScudoWrappersCTest, OtherAlloc) {
  const size_t PageSize = sysconf(_SC_PAGESIZE);

  void *P = pvalloc(Size);
  EXPECT_NE(P, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(P) & (PageSize - 1), 0U);
  EXPECT_LE(PageSize, malloc_usable_size(P));
  free(P);

  EXPECT_EQ(pvalloc(SIZE_MAX), nullptr);

  P = pvalloc(Size);
  EXPECT_NE(P, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(P) & (PageSize - 1), 0U);
  free(P);

  EXPECT_EQ(valloc(SIZE_MAX), nullptr);
}
