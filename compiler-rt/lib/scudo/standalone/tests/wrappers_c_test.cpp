//===-- wrappers_c_test.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "scudo/interface.h"
#include "tests/scudo_unit_test.h"

#include <errno.h>
#include <limits.h>
#include <malloc.h>
#include <stdlib.h>
#include <unistd.h>

extern "C" {
void malloc_enable(void);
void malloc_disable(void);
int malloc_iterate(uintptr_t base, size_t size,
                   void (*callback)(uintptr_t base, size_t size, void *arg),
                   void *arg);
void *valloc(size_t size);
void *pvalloc(size_t size);
}

// Note that every C allocation function in the test binary will be fulfilled
// by Scudo (this includes the gtest APIs, etc.), which is a test by itself.
// But this might also lead to unexpected side-effects, since the allocation and
// deallocation operations in the TEST functions will coexist with others (see
// the EXPECT_DEATH comment below).

// We have to use a small quarantine to make sure that our double-free tests
// trigger. Otherwise EXPECT_DEATH ends up reallocating the chunk that was just
// freed (this depends on the size obviously) and the following free succeeds.

static const size_t Size = 100U;

TEST(ScudoWrappersCTest, Malloc) {
  void *P = malloc(Size);
  EXPECT_NE(P, nullptr);
  EXPECT_LE(Size, malloc_usable_size(P));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(P) % FIRST_32_SECOND_64(8U, 16U), 0U);

  // An update to this warning in Clang now triggers in this line, but it's ok
  // because the check is expecting a bad pointer and should fail.
#if defined(__has_warning) && __has_warning("-Wfree-nonheap-object")
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfree-nonheap-object"
#endif
  EXPECT_DEATH(
      free(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(P) | 1U)), "");
#if defined(__has_warning) && __has_warning("-Wfree-nonheap-object")
#pragma GCC diagnostic pop
#endif

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

#if !SCUDO_FUCHSIA
TEST(ScudoWrappersCTest, MallOpt) {
  errno = 0;
  EXPECT_EQ(mallopt(-1000, 1), 0);
  // mallopt doesn't set errno.
  EXPECT_EQ(errno, 0);

  EXPECT_EQ(mallopt(M_PURGE, 0), 1);

  EXPECT_EQ(mallopt(M_DECAY_TIME, 1), 1);
  EXPECT_EQ(mallopt(M_DECAY_TIME, 0), 1);
  EXPECT_EQ(mallopt(M_DECAY_TIME, 1), 1);
  EXPECT_EQ(mallopt(M_DECAY_TIME, 0), 1);

  if (SCUDO_ANDROID) {
    EXPECT_EQ(mallopt(M_CACHE_COUNT_MAX, 100), 1);
    EXPECT_EQ(mallopt(M_CACHE_SIZE_MAX, 1024 * 1024 * 2), 1);
    EXPECT_EQ(mallopt(M_TSDS_COUNT_MAX, 10), 1);
  }
}
#endif

TEST(ScudoWrappersCTest, OtherAlloc) {
#if !SCUDO_FUCHSIA
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
#endif

  EXPECT_EQ(valloc(SIZE_MAX), nullptr);
}

#if !SCUDO_FUCHSIA
TEST(ScudoWrappersCTest, MallInfo) {
  const size_t BypassQuarantineSize = 1024U;

  struct mallinfo MI = mallinfo();
  size_t Allocated = MI.uordblks;
  void *P = malloc(BypassQuarantineSize);
  EXPECT_NE(P, nullptr);
  MI = mallinfo();
  EXPECT_GE(static_cast<size_t>(MI.uordblks), Allocated + BypassQuarantineSize);
  EXPECT_GT(static_cast<size_t>(MI.hblkhd), 0U);
  size_t Free = MI.fordblks;
  free(P);
  MI = mallinfo();
  EXPECT_GE(static_cast<size_t>(MI.fordblks), Free + BypassQuarantineSize);
}
#endif

static uintptr_t BoundaryP;
static size_t Count;

static void callback(uintptr_t Base, size_t Size, void *Arg) {
  if (Base == BoundaryP)
    Count++;
}

// Verify that a block located on an iteration boundary is not mis-accounted.
// To achieve this, we allocate a chunk for which the backing block will be
// aligned on a page, then run the malloc_iterate on both the pages that the
// block is a boundary for. It must only be seen once by the callback function.
TEST(ScudoWrappersCTest, MallocIterateBoundary) {
  const size_t PageSize = sysconf(_SC_PAGESIZE);
  const size_t BlockDelta = FIRST_32_SECOND_64(8U, 16U);
  const size_t SpecialSize = PageSize - BlockDelta;

  // We aren't guaranteed that any size class is exactly a page wide. So we need
  // to keep making allocations until we succeed.
  //
  // With a 16-byte block alignment and 4096-byte page size, each allocation has
  // a probability of (1 - (16/4096)) of failing to meet the alignment
  // requirements, and the probability of failing 65536 times is
  // (1 - (16/4096))^65536 < 10^-112. So if we still haven't succeeded after
  // 65536 tries, give up.
  uintptr_t Block;
  void *P = nullptr;
  for (unsigned I = 0; I != 65536; ++I) {
    void *PrevP = P;
    P = malloc(SpecialSize);
    EXPECT_NE(P, nullptr);
    *reinterpret_cast<void **>(P) = PrevP;
    BoundaryP = reinterpret_cast<uintptr_t>(P);
    Block = BoundaryP - BlockDelta;
    if ((Block & (PageSize - 1)) == 0U)
      break;
  }
  EXPECT_EQ((Block & (PageSize - 1)), 0U);

  Count = 0U;
  malloc_disable();
  malloc_iterate(Block - PageSize, PageSize, callback, nullptr);
  malloc_iterate(Block, PageSize, callback, nullptr);
  malloc_enable();
  EXPECT_EQ(Count, 1U);

  while (P) {
    void *NextP = *reinterpret_cast<void **>(P);
    free(P);
    P = NextP;
  }
}

// Fuchsia doesn't have alarm, fork or malloc_info.
#if !SCUDO_FUCHSIA
TEST(ScudoWrappersCTest, MallocDisableDeadlock) {
  // We expect heap operations within a disable/enable scope to deadlock.
  EXPECT_DEATH(
      {
        void *P = malloc(Size);
        EXPECT_NE(P, nullptr);
        free(P);
        malloc_disable();
        alarm(1);
        P = malloc(Size);
        malloc_enable();
      },
      "");
}

TEST(ScudoWrappersCTest, MallocInfo) {
  // Use volatile so that the allocations don't get optimized away.
  void *volatile P1 = malloc(1234);
  void *volatile P2 = malloc(4321);

  char Buffer[16384];
  FILE *F = fmemopen(Buffer, sizeof(Buffer), "w+");
  EXPECT_NE(F, nullptr);
  errno = 0;
  EXPECT_EQ(malloc_info(0, F), 0);
  EXPECT_EQ(errno, 0);
  fclose(F);
  EXPECT_EQ(strncmp(Buffer, "<malloc version=\"scudo-", 23), 0);
  EXPECT_NE(nullptr, strstr(Buffer, "<alloc size=\"1234\" count=\""));
  EXPECT_NE(nullptr, strstr(Buffer, "<alloc size=\"4321\" count=\""));

  free(P1);
  free(P2);
}

TEST(ScudoWrappersCTest, Fork) {
  void *P;
  pid_t Pid = fork();
  EXPECT_GE(Pid, 0);
  if (Pid == 0) {
    P = malloc(Size);
    EXPECT_NE(P, nullptr);
    memset(P, 0x42, Size);
    free(P);
    _exit(0);
  }
  waitpid(Pid, nullptr, 0);
  P = malloc(Size);
  EXPECT_NE(P, nullptr);
  memset(P, 0x42, Size);
  free(P);

  // fork should stall if the allocator has been disabled.
  EXPECT_DEATH(
      {
        malloc_disable();
        alarm(1);
        Pid = fork();
        EXPECT_GE(Pid, 0);
      },
      "");
}

static pthread_mutex_t Mutex;
static pthread_cond_t Conditional = PTHREAD_COND_INITIALIZER;
static bool Ready;

static void *enableMalloc(void *Unused) {
  // Initialize the allocator for this thread.
  void *P = malloc(Size);
  EXPECT_NE(P, nullptr);
  memset(P, 0x42, Size);
  free(P);

  // Signal the main thread we are ready.
  pthread_mutex_lock(&Mutex);
  Ready = true;
  pthread_cond_signal(&Conditional);
  pthread_mutex_unlock(&Mutex);

  // Wait for the malloc_disable & fork, then enable the allocator again.
  sleep(1);
  malloc_enable();

  return nullptr;
}

TEST(ScudoWrappersCTest, DisableForkEnable) {
  pthread_t ThreadId;
  Ready = false;
  EXPECT_EQ(pthread_create(&ThreadId, nullptr, &enableMalloc, nullptr), 0);

  // Wait for the thread to be warmed up.
  pthread_mutex_lock(&Mutex);
  while (!Ready)
    pthread_cond_wait(&Conditional, &Mutex);
  pthread_mutex_unlock(&Mutex);

  // Disable the allocator and fork. fork should succeed after malloc_enable.
  malloc_disable();
  pid_t Pid = fork();
  EXPECT_GE(Pid, 0);
  if (Pid == 0) {
    void *P = malloc(Size);
    EXPECT_NE(P, nullptr);
    memset(P, 0x42, Size);
    free(P);
    _exit(0);
  }
  waitpid(Pid, nullptr, 0);
  EXPECT_EQ(pthread_join(ThreadId, 0), 0);
}

#endif // SCUDO_FUCHSIA
