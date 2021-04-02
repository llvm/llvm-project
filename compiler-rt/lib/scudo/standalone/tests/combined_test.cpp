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
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

static std::mutex Mutex;
static std::condition_variable Cv;
static bool Ready;

static constexpr scudo::Chunk::Origin Origin = scudo::Chunk::Origin::Malloc;

template <class Config> struct UseQuarantineSetter {
  UseQuarantineSetter(bool Value) { UseQuarantine = Value; }
  ~UseQuarantineSetter() { UseQuarantine = true; }
};

// Fuchsia complains that the function is not used.
UNUSED static void disableDebuggerdMaybe() {
#if SCUDO_ANDROID
  // Disable the debuggerd signal handler on Android, without this we can end
  // up spending a significant amount of time creating tombstones.
  signal(SIGSEGV, SIG_DFL);
#endif
}

template <class AllocatorT>
bool isPrimaryAllocation(scudo::uptr Size, scudo::uptr Alignment) {
  const scudo::uptr MinAlignment = 1UL << SCUDO_MIN_ALIGNMENT_LOG;
  if (Alignment < MinAlignment)
    Alignment = MinAlignment;
  const scudo::uptr NeededSize =
      scudo::roundUpTo(Size, MinAlignment) +
      ((Alignment > MinAlignment) ? Alignment : scudo::Chunk::getHeaderSize());
  return AllocatorT::PrimaryT::canAllocate(NeededSize);
}

template <class AllocatorT>
void checkMemoryTaggingMaybe(AllocatorT *Allocator, void *P, scudo::uptr Size,
                             scudo::uptr Alignment) {
  const scudo::uptr MinAlignment = 1UL << SCUDO_MIN_ALIGNMENT_LOG;
  Size = scudo::roundUpTo(Size, MinAlignment);
  if (Allocator->useMemoryTaggingTestOnly())
    EXPECT_DEATH(
        {
          disableDebuggerdMaybe();
          reinterpret_cast<char *>(P)[-1] = 0xaa;
        },
        "");
  if (isPrimaryAllocation<AllocatorT>(Size, Alignment)
          ? Allocator->useMemoryTaggingTestOnly()
          : Alignment == MinAlignment) {
    EXPECT_DEATH(
        {
          disableDebuggerdMaybe();
          reinterpret_cast<char *>(P)[Size] = 0xaa;
        },
        "");
  }
}

template <typename Config> struct TestAllocator : scudo::Allocator<Config> {
  TestAllocator() {
    this->reset();
    this->initThreadMaybe();
    if (scudo::archSupportsMemoryTagging() &&
        !scudo::systemDetectsMemoryTagFaultsTestOnly())
      this->disableMemoryTagging();
  }
  ~TestAllocator() { this->unmapTestOnly(); }
};

template <class Config> static void testAllocator() {
  UseQuarantineSetter<Config> MUQ(
      std::is_same<Config, scudo::AndroidConfig>::value);
  using AllocatorT = TestAllocator<Config>;
  auto Allocator = std::unique_ptr<AllocatorT>(new AllocatorT());

  static scudo::u8 StaticBuffer[scudo::Chunk::getHeaderSize() + 1];
  EXPECT_FALSE(
      Allocator->isOwned(&StaticBuffer[scudo::Chunk::getHeaderSize()]));

  scudo::u8 StackBuffer[scudo::Chunk::getHeaderSize() + 1];
  for (scudo::uptr I = 0; I < sizeof(StackBuffer); I++)
    StackBuffer[I] = 0x42U;
  EXPECT_FALSE(Allocator->isOwned(&StackBuffer[scudo::Chunk::getHeaderSize()]));
  for (scudo::uptr I = 0; I < sizeof(StackBuffer); I++)
    EXPECT_EQ(StackBuffer[I], 0x42U);

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
        checkMemoryTaggingMaybe(Allocator.get(), P, Size, Align);
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
        ASSERT_EQ((reinterpret_cast<char *>(P))[I], 0);
      memset(P, 0xaa, Size);
      Allocator->deallocate(P, Origin, Size);
    }
  }
  Allocator->releaseToOS();

  // Ensure that specifying ZeroContents returns a zero'd out block.
  Allocator->setFillContents(scudo::ZeroFill);
  for (scudo::uptr SizeLog = 0U; SizeLog <= 20U; SizeLog++) {
    for (scudo::uptr Delta = 0U; Delta <= 4U; Delta++) {
      const scudo::uptr Size = (1U << SizeLog) + Delta * 128U;
      void *P = Allocator->allocate(Size, Origin, 1U << MinAlignLog, false);
      EXPECT_NE(P, nullptr);
      for (scudo::uptr I = 0; I < Size; I++)
        ASSERT_EQ((reinterpret_cast<char *>(P))[I], 0);
      memset(P, 0xaa, Size);
      Allocator->deallocate(P, Origin, Size);
    }
  }
  Allocator->releaseToOS();

  // Ensure that specifying PatternOrZeroFill returns a pattern or zero filled
  // block. The primary allocator only produces pattern filled blocks if MTE
  // is disabled, so we only require pattern filled blocks in that case.
  Allocator->setFillContents(scudo::PatternOrZeroFill);
  for (scudo::uptr SizeLog = 0U; SizeLog <= 20U; SizeLog++) {
    for (scudo::uptr Delta = 0U; Delta <= 4U; Delta++) {
      const scudo::uptr Size = (1U << SizeLog) + Delta * 128U;
      void *P = Allocator->allocate(Size, Origin, 1U << MinAlignLog, false);
      EXPECT_NE(P, nullptr);
      for (scudo::uptr I = 0; I < Size; I++) {
        unsigned char V = (reinterpret_cast<unsigned char *>(P))[I];
        if (isPrimaryAllocation<AllocatorT>(Size, 1U << MinAlignLog) &&
            !Allocator->useMemoryTaggingTestOnly())
          ASSERT_EQ(V, scudo::PatternFillByte);
        else
          ASSERT_TRUE(V == scudo::PatternFillByte || V == 0);
      }
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
    if (Allocator->getHeaderTaggedPointer(P) ==
        Allocator->getHeaderTaggedPointer(NeedleP))
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
  // the same block size, but that should be the case for MaxSize - 64 with our
  // default class size maps.
  constexpr scudo::uptr ReallocSize = MaxSize - 64;
  P = Allocator->allocate(ReallocSize, Origin);
  memset(P, Marker, ReallocSize);
  for (scudo::sptr Delta = -32; Delta < 32; Delta += 8) {
    const scudo::uptr NewSize = ReallocSize + Delta;
    void *NewP = Allocator->reallocate(P, NewSize);
    EXPECT_EQ(NewP, P);
    for (scudo::uptr I = 0; I < ReallocSize - 32; I++)
      EXPECT_EQ((reinterpret_cast<char *>(NewP))[I], Marker);
    checkMemoryTaggingMaybe(Allocator.get(), NewP, NewSize, 0);
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

  // Check that use-after-free is detected.
  for (scudo::uptr SizeLog = 0U; SizeLog <= 20U; SizeLog++) {
    const scudo::uptr Size = 1U << SizeLog;
    if (!Allocator->useMemoryTaggingTestOnly())
      continue;
    EXPECT_DEATH(
        {
          disableDebuggerdMaybe();
          void *P = Allocator->allocate(Size, Origin);
          Allocator->deallocate(P, Origin);
          reinterpret_cast<char *>(P)[0] = 0xaa;
        },
        "");
    EXPECT_DEATH(
        {
          disableDebuggerdMaybe();
          void *P = Allocator->allocate(Size, Origin);
          Allocator->deallocate(P, Origin);
          reinterpret_cast<char *>(P)[Size - 1] = 0xaa;
        },
        "");
  }

  if (Allocator->useMemoryTaggingTestOnly()) {
    // Check that disabling memory tagging works correctly.
    void *P = Allocator->allocate(2048, Origin);
    EXPECT_DEATH(reinterpret_cast<char *>(P)[2048] = 0xaa, "");
    scudo::disableMemoryTagChecksTestOnly();
    Allocator->disableMemoryTagging();
    reinterpret_cast<char *>(P)[2048] = 0xaa;
    Allocator->deallocate(P, Origin);

    P = Allocator->allocate(2048, Origin);
    EXPECT_EQ(scudo::untagPointer(P), P);
    reinterpret_cast<char *>(P)[2048] = 0xaa;
    Allocator->deallocate(P, Origin);

    Allocator->releaseToOS();

    // Disabling memory tag checks may interfere with subsequent tests.
    // Re-enable them now.
    scudo::enableMemoryTagChecksTestOnly();
  }

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

  bool UnlockRequired;
  auto *TSD = Allocator->getTSDRegistry()->getTSDAndLock(&UnlockRequired);
  EXPECT_TRUE(!TSD->Cache.isEmpty());
  TSD->Cache.drain();
  EXPECT_TRUE(TSD->Cache.isEmpty());
  if (UnlockRequired)
    TSD->unlock();
}

// Test that multiple instantiations of the allocator have not messed up the
// process's signal handlers (GWP-ASan used to do this).
void testSEGV() {
  const scudo::uptr Size = 4 * scudo::getPageSizeCached();
  scudo::MapPlatformData Data = {};
  void *P = scudo::map(nullptr, Size, "testSEGV", MAP_NOACCESS, &Data);
  EXPECT_NE(P, nullptr);
  EXPECT_DEATH(memset(P, 0xaa, Size), "");
  scudo::unmap(P, Size, UNMAP_ALL, &Data);
}

TEST(ScudoCombinedTest, BasicCombined) {
  testAllocator<scudo::AndroidSvelteConfig>();
#if SCUDO_FUCHSIA
  testAllocator<scudo::FuchsiaConfig>();
#else
  testAllocator<scudo::DefaultConfig>();
  testAllocator<scudo::AndroidConfig>();
  testSEGV();
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
  Ready = false;
  using AllocatorT = TestAllocator<Config>;
  auto Allocator = std::unique_ptr<AllocatorT>(new AllocatorT());
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
  testAllocatorThreaded<scudo::AndroidSvelteConfig>();
#if SCUDO_FUCHSIA
  testAllocatorThreaded<scudo::FuchsiaConfig>();
#else
  testAllocatorThreaded<scudo::DefaultConfig>();
  testAllocatorThreaded<scudo::AndroidConfig>();
#endif
}

struct DeathSizeClassConfig {
  static const scudo::uptr NumBits = 1;
  static const scudo::uptr MinSizeLog = 10;
  static const scudo::uptr MidSizeLog = 10;
  static const scudo::uptr MaxSizeLog = 13;
  static const scudo::u32 MaxNumCachedHint = 4;
  static const scudo::uptr MaxBytesCachedLog = 12;
};

static const scudo::uptr DeathRegionSizeLog = 20U;
struct DeathConfig {
  static const bool MaySupportMemoryTagging = false;

  // Tiny allocator, its Primary only serves chunks of four sizes.
  using SizeClassMap = scudo::FixedSizeClassMap<DeathSizeClassConfig>;
  typedef scudo::SizeClassAllocator64<DeathConfig> Primary;
  static const scudo::uptr PrimaryRegionSizeLog = DeathRegionSizeLog;
  static const scudo::s32 PrimaryMinReleaseToOsIntervalMs = INT32_MIN;
  static const scudo::s32 PrimaryMaxReleaseToOsIntervalMs = INT32_MAX;
  typedef scudo::uptr PrimaryCompactPtrT;
  static const scudo::uptr PrimaryCompactPtrScale = 0;

  typedef scudo::MapAllocatorNoCache SecondaryCache;
  template <class A> using TSDRegistryT = scudo::TSDRegistrySharedT<A, 1U, 1U>;
};

TEST(ScudoCombinedTest, DeathCombined) {
  using AllocatorT = TestAllocator<DeathConfig>;
  auto Allocator = std::unique_ptr<AllocatorT>(new AllocatorT());

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
  using AllocatorT = TestAllocator<DeathConfig>;
  auto Allocator = std::unique_ptr<AllocatorT>(new AllocatorT());

  Allocator->releaseToOS();
}

// Verify that when a region gets full, the allocator will still manage to
// fulfill the allocation through a larger size class.
TEST(ScudoCombinedTest, FullRegion) {
  using AllocatorT = TestAllocator<DeathConfig>;
  auto Allocator = std::unique_ptr<AllocatorT>(new AllocatorT());

  std::vector<void *> V;
  scudo::uptr FailedAllocationsCount = 0;
  for (scudo::uptr ClassId = 1U;
       ClassId <= DeathConfig::SizeClassMap::LargestClassId; ClassId++) {
    const scudo::uptr Size =
        DeathConfig::SizeClassMap::getSizeByClassId(ClassId);
    // Allocate enough to fill all of the regions above this one.
    const scudo::uptr MaxNumberOfChunks =
        ((1U << DeathRegionSizeLog) / Size) *
        (DeathConfig::SizeClassMap::LargestClassId - ClassId + 1);
    void *P;
    for (scudo::uptr I = 0; I <= MaxNumberOfChunks; I++) {
      P = Allocator->allocate(Size - 64U, Origin);
      if (!P)
        FailedAllocationsCount++;
      else
        V.push_back(P);
    }
    while (!V.empty()) {
      Allocator->deallocate(V.back(), Origin);
      V.pop_back();
    }
  }
  EXPECT_EQ(FailedAllocationsCount, 0U);
}

TEST(ScudoCombinedTest, OddEven) {
  using AllocatorT = TestAllocator<scudo::AndroidConfig>;
  using SizeClassMap = AllocatorT::PrimaryT::SizeClassMap;
  auto Allocator = std::unique_ptr<AllocatorT>(new AllocatorT());

  if (!Allocator->useMemoryTaggingTestOnly())
    return;

  auto CheckOddEven = [](scudo::uptr P1, scudo::uptr P2) {
    scudo::uptr Tag1 = scudo::extractTag(scudo::loadTag(P1));
    scudo::uptr Tag2 = scudo::extractTag(scudo::loadTag(P2));
    EXPECT_NE(Tag1 % 2, Tag2 % 2);
  };

  for (scudo::uptr ClassId = 1U; ClassId <= SizeClassMap::LargestClassId;
       ClassId++) {
    const scudo::uptr Size = SizeClassMap::getSizeByClassId(ClassId);

    std::set<scudo::uptr> Ptrs;
    bool Found = false;
    for (unsigned I = 0; I != 65536; ++I) {
      scudo::uptr P = scudo::untagPointer(reinterpret_cast<scudo::uptr>(
          Allocator->allocate(Size - scudo::Chunk::getHeaderSize(), Origin)));
      if (Ptrs.count(P - Size)) {
        Found = true;
        CheckOddEven(P, P - Size);
        break;
      }
      if (Ptrs.count(P + Size)) {
        Found = true;
        CheckOddEven(P, P + Size);
        break;
      }
      Ptrs.insert(P);
    }
    EXPECT_TRUE(Found);
  }
}

TEST(ScudoCombinedTest, DisableMemInit) {
  using AllocatorT = TestAllocator<scudo::AndroidConfig>;
  using SizeClassMap = AllocatorT::PrimaryT::SizeClassMap;
  auto Allocator = std::unique_ptr<AllocatorT>(new AllocatorT());

  std::vector<void *> Ptrs(65536, nullptr);

  Allocator->setOption(scudo::Option::ThreadDisableMemInit, 1);

  constexpr scudo::uptr MinAlignLog = FIRST_32_SECOND_64(3U, 4U);

  // Test that if mem-init is disabled on a thread, calloc should still work as
  // expected. This is tricky to ensure when MTE is enabled, so this test tries
  // to exercise the relevant code on our MTE path.
  for (scudo::uptr ClassId = 1U; ClassId <= 8; ClassId++) {
    const scudo::uptr Size =
        SizeClassMap::getSizeByClassId(ClassId) - scudo::Chunk::getHeaderSize();
    if (Size < 8)
      continue;
    for (unsigned I = 0; I != Ptrs.size(); ++I) {
      Ptrs[I] = Allocator->allocate(Size, Origin);
      memset(Ptrs[I], 0xaa, Size);
    }
    for (unsigned I = 0; I != Ptrs.size(); ++I)
      Allocator->deallocate(Ptrs[I], Origin, Size);
    for (unsigned I = 0; I != Ptrs.size(); ++I) {
      Ptrs[I] = Allocator->allocate(Size - 8, Origin);
      memset(Ptrs[I], 0xbb, Size - 8);
    }
    for (unsigned I = 0; I != Ptrs.size(); ++I)
      Allocator->deallocate(Ptrs[I], Origin, Size - 8);
    for (unsigned I = 0; I != Ptrs.size(); ++I) {
      Ptrs[I] = Allocator->allocate(Size, Origin, 1U << MinAlignLog, true);
      for (scudo::uptr J = 0; J < Size; ++J)
        ASSERT_EQ((reinterpret_cast<char *>(Ptrs[I]))[J], 0);
    }
  }

  Allocator->setOption(scudo::Option::ThreadDisableMemInit, 0);
}
