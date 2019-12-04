//===-- combined_test.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "tests/scudo_unit_test.h"

#include "allocator_config.h"
#include "combined.h"

#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

static std::mutex Mutex;
static std::condition_variable Cv;
static bool Ready = false;

static constexpr scudo::Chunk::Origin Origin = scudo::Chunk::Origin::Malloc;

template <class Config> static void testAllocator() {
  using AllocatorT = scudo::Allocator<Config>;
  auto Deleter = [](AllocatorT *A) {
    A->unmapTestOnly();
    delete A;
  };
  std::unique_ptr<AllocatorT, decltype(Deleter)> Allocator(new AllocatorT,
                                                           Deleter);
  Allocator->reset();

  EXPECT_FALSE(Allocator->isOwned(&Mutex));
  EXPECT_FALSE(Allocator->isOwned(&Allocator));
  scudo::u64 StackVariable = 0x42424242U;
  EXPECT_FALSE(Allocator->isOwned(&StackVariable));
  EXPECT_EQ(StackVariable, 0x42424242U);

  constexpr scudo::uptr MinAlignLog = FIRST_32_SECOND_64(3U, 4U);

  // This allocates and deallocates a bunch of chunks, with a wide range of
  // sizes and alignments, with a focus on sizes that could trigger weird
  // behaviors (plus or minus a small delta of a power of two for example).
  for (scudo::uptr SizeLog = 0U; SizeLog <= 20U; SizeLog++) {
    for (scudo::uptr AlignLog = MinAlignLog; AlignLog <= 16U; AlignLog++) {
      const scudo::uptr Align = 1U << AlignLog;
      for (scudo::sptr Delta = -32; Delta <= 32; Delta++) {
        if (static_cast<scudo::sptr>(1U << SizeLog) + Delta <= 0)
          continue;
        const scudo::uptr Size = (1U << SizeLog) + Delta;
        void *P = Allocator->allocate(Size, Origin, Align);
        EXPECT_NE(P, nullptr);
        EXPECT_TRUE(Allocator->isOwned(P));
        EXPECT_TRUE(scudo::isAligned(reinterpret_cast<scudo::uptr>(P), Align));
        EXPECT_LE(Size, Allocator->getUsableSize(P));
        memset(P, 0xaa, Size);
        Allocator->deallocate(P, Origin, Size);
      }
    }
  }
  Allocator->releaseToOS();

  // Ensure that specifying ZeroContents returns a zero'd out block.
  for (scudo::uptr SizeLog = 0U; SizeLog <= 20U; SizeLog++) {
    for (scudo::uptr Delta = 0U; Delta <= 4U; Delta++) {
      const scudo::uptr Size = (1U << SizeLog) + Delta * 128U;
      void *P = Allocator->allocate(Size, Origin, 1U << MinAlignLog, true);
      EXPECT_NE(P, nullptr);
      for (scudo::uptr I = 0; I < Size; I++)
        EXPECT_EQ((reinterpret_cast<char *>(P))[I], 0);
      memset(P, 0xaa, Size);
      Allocator->deallocate(P, Origin, Size);
    }
  }
  Allocator->releaseToOS();

  // Verify that a chunk will end up being reused, at some point.
  const scudo::uptr NeedleSize = 1024U;
  void *NeedleP = Allocator->allocate(NeedleSize, Origin);
  Allocator->deallocate(NeedleP, Origin);
  bool Found = false;
  for (scudo::uptr I = 0; I < 1024U && !Found; I++) {
    void *P = Allocator->allocate(NeedleSize, Origin);
    if (P == NeedleP)
      Found = true;
    Allocator->deallocate(P, Origin);
  }
  EXPECT_TRUE(Found);

  constexpr scudo::uptr MaxSize = Config::Primary::SizeClassMap::MaxSize;

  // Reallocate a large chunk all the way down to a byte, verifying that we
  // preserve the data in the process.
  scudo::uptr Size = MaxSize * 2;
  const scudo::uptr DataSize = 2048U;
  void *P = Allocator->allocate(Size, Origin);
  const char Marker = 0xab;
  memset(P, Marker, scudo::Min(Size, DataSize));
  while (Size > 1U) {
    Size /= 2U;
    void *NewP = Allocator->reallocate(P, Size);
    EXPECT_NE(NewP, nullptr);
    for (scudo::uptr J = 0; J < scudo::Min(Size, DataSize); J++)
      EXPECT_EQ((reinterpret_cast<char *>(NewP))[J], Marker);
    P = NewP;
  }
  Allocator->deallocate(P, Origin);

  // Check that reallocating a chunk to a slightly smaller or larger size
  // returns the same chunk. This requires that all the sizes we iterate on use
  // the same block size, but that should be the case for 2048 with our default
  // class size maps.
  P = Allocator->allocate(DataSize, Origin);
  memset(P, Marker, DataSize);
  for (scudo::sptr Delta = -32; Delta < 32; Delta += 8) {
    const scudo::uptr NewSize = DataSize + Delta;
    void *NewP = Allocator->reallocate(P, NewSize);
    EXPECT_EQ(NewP, P);
    for (scudo::uptr I = 0; I < DataSize - 32; I++)
      EXPECT_EQ((reinterpret_cast<char *>(NewP))[I], Marker);
  }
  Allocator->deallocate(P, Origin);

  // Allocates a bunch of chunks, then iterate over all the chunks, ensuring
  // they are the ones we allocated. This requires the allocator to not have any
  // other allocated chunk at this point (eg: won't work with the Quarantine).
  if (!UseQuarantine) {
    std::vector<void *> V;
    for (scudo::uptr I = 0; I < 64U; I++)
      V.push_back(Allocator->allocate(rand() % (MaxSize / 2U), Origin));
    Allocator->disable();
    Allocator->iterateOverChunks(
        0U, static_cast<scudo::uptr>(SCUDO_MMAP_RANGE_SIZE - 1),
        [](uintptr_t Base, size_t Size, void *Arg) {
          std::vector<void *> *V = reinterpret_cast<std::vector<void *> *>(Arg);
          void *P = reinterpret_cast<void *>(Base);
          EXPECT_NE(std::find(V->begin(), V->end(), P), V->end());
        },
        reinterpret_cast<void *>(&V));
    Allocator->enable();
    while (!V.empty()) {
      Allocator->deallocate(V.back(), Origin);
      V.pop_back();
    }
  }

  Allocator->releaseToOS();

  scudo::uptr BufferSize = 8192;
  std::vector<char> Buffer(BufferSize);
  scudo::uptr ActualSize = Allocator->getStats(Buffer.data(), BufferSize);
  while (ActualSize > BufferSize) {
    BufferSize = ActualSize + 1024;
    Buffer.resize(BufferSize);
    ActualSize = Allocator->getStats(Buffer.data(), BufferSize);
  }
  std::string Stats(Buffer.begin(), Buffer.end());
  // Basic checks on the contents of the statistics output, which also allows us
  // to verify that we got it all.
  EXPECT_NE(Stats.find("Stats: SizeClassAllocator"), std::string::npos);
  EXPECT_NE(Stats.find("Stats: MapAllocator"), std::string::npos);
  EXPECT_NE(Stats.find("Stats: Quarantine"), std::string::npos);
}

TEST(ScudoCombinedTest, BasicCombined) {
  UseQuarantine = false;
  testAllocator<scudo::AndroidSvelteConfig>();
#if SCUDO_FUCHSIA
  testAllocator<scudo::FuchsiaConfig>();
#else
  testAllocator<scudo::DefaultConfig>();
  UseQuarantine = true;
  testAllocator<scudo::AndroidConfig>();
#endif
}

template <typename AllocatorT> static void stressAllocator(AllocatorT *A) {
  {
    std::unique_lock<std::mutex> Lock(Mutex);
    while (!Ready)
      Cv.wait(Lock);
  }
  std::vector<std::pair<void *, scudo::uptr>> V;
  for (scudo::uptr I = 0; I < 256U; I++) {
    const scudo::uptr Size = std::rand() % 4096U;
    void *P = A->allocate(Size, Origin);
    // A region could have ran out of memory, resulting in a null P.
    if (P)
      V.push_back(std::make_pair(P, Size));
  }
  while (!V.empty()) {
    auto Pair = V.back();
    A->deallocate(Pair.first, Origin, Pair.second);
    V.pop_back();
  }
}

template <class Config> static void testAllocatorThreaded() {
  using AllocatorT = scudo::Allocator<Config>;
  auto Deleter = [](AllocatorT *A) {
    A->unmapTestOnly();
    delete A;
  };
  std::unique_ptr<AllocatorT, decltype(Deleter)> Allocator(new AllocatorT,
                                                           Deleter);
  Allocator->reset();
  std::thread Threads[32];
  for (scudo::uptr I = 0; I < ARRAY_SIZE(Threads); I++)
    Threads[I] = std::thread(stressAllocator<AllocatorT>, Allocator.get());
  {
    std::unique_lock<std::mutex> Lock(Mutex);
    Ready = true;
    Cv.notify_all();
  }
  for (auto &T : Threads)
    T.join();
  Allocator->releaseToOS();
}

TEST(ScudoCombinedTest, ThreadedCombined) {
  UseQuarantine = false;
  testAllocatorThreaded<scudo::AndroidSvelteConfig>();
#if SCUDO_FUCHSIA
  testAllocatorThreaded<scudo::FuchsiaConfig>();
#else
  testAllocatorThreaded<scudo::DefaultConfig>();
  UseQuarantine = true;
  testAllocatorThreaded<scudo::AndroidConfig>();
#endif
}

struct DeathConfig {
  // Tiny allocator, its Primary only serves chunks of 1024 bytes.
  using DeathSizeClassMap = scudo::SizeClassMap<1U, 10U, 10U, 10U, 1U, 10U>;
  typedef scudo::SizeClassAllocator64<DeathSizeClassMap, 20U> Primary;
  typedef scudo::MapAllocator<0U> Secondary;
  template <class A> using TSDRegistryT = scudo::TSDRegistrySharedT<A, 1U>;
};

TEST(ScudoCombinedTest, DeathCombined) {
  using AllocatorT = scudo::Allocator<DeathConfig>;
  auto Deleter = [](AllocatorT *A) {
    A->unmapTestOnly();
    delete A;
  };
  std::unique_ptr<AllocatorT, decltype(Deleter)> Allocator(new AllocatorT,
                                                           Deleter);
  Allocator->reset();

  const scudo::uptr Size = 1000U;
  void *P = Allocator->allocate(Size, Origin);
  EXPECT_NE(P, nullptr);

  // Invalid sized deallocation.
  EXPECT_DEATH(Allocator->deallocate(P, Origin, Size + 8U), "");

  // Misaligned pointer. Potentially unused if EXPECT_DEATH isn't available.
  UNUSED void *MisalignedP =
      reinterpret_cast<void *>(reinterpret_cast<scudo::uptr>(P) | 1U);
  EXPECT_DEATH(Allocator->deallocate(MisalignedP, Origin, Size), "");
  EXPECT_DEATH(Allocator->reallocate(MisalignedP, Size * 2U), "");

  // Header corruption.
  scudo::u64 *H =
      reinterpret_cast<scudo::u64 *>(scudo::Chunk::getAtomicHeader(P));
  *H ^= 0x42U;
  EXPECT_DEATH(Allocator->deallocate(P, Origin, Size), "");
  *H ^= 0x420042U;
  EXPECT_DEATH(Allocator->deallocate(P, Origin, Size), "");
  *H ^= 0x420000U;

  // Invalid chunk state.
  Allocator->deallocate(P, Origin, Size);
  EXPECT_DEATH(Allocator->deallocate(P, Origin, Size), "");
  EXPECT_DEATH(Allocator->reallocate(P, Size * 2U), "");
  EXPECT_DEATH(Allocator->getUsableSize(P), "");
}

// Ensure that releaseToOS can be called prior to any other allocator
// operation without issue.
TEST(ScudoCombinedTest, ReleaseToOS) {
  using AllocatorT = scudo::Allocator<DeathConfig>;
  auto Deleter = [](AllocatorT *A) {
    A->unmapTestOnly();
    delete A;
  };
  std::unique_ptr<AllocatorT, decltype(Deleter)> Allocator(new AllocatorT,
                                                           Deleter);
  Allocator->reset();

  Allocator->releaseToOS();
}
