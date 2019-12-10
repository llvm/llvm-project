//===-- combined.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCUDO_COMBINED_H_
#define SCUDO_COMBINED_H_

#include "chunk.h"
#include "common.h"
#include "flags.h"
#include "flags_parser.h"
#include "interface.h"
#include "local_cache.h"
#include "quarantine.h"
#include "report.h"
#include "secondary.h"
#include "tsd.h"

namespace scudo {

template <class Params> class Allocator {
public:
  using PrimaryT = typename Params::Primary;
  using CacheT = typename PrimaryT::CacheT;
  typedef Allocator<Params> ThisT;
  typedef typename Params::template TSDRegistryT<ThisT> TSDRegistryT;

  struct QuarantineCallback {
    explicit QuarantineCallback(ThisT &Instance, CacheT &LocalCache)
        : Allocator(Instance), Cache(LocalCache) {}

    // Chunk recycling function, returns a quarantined chunk to the backend,
    // first making sure it hasn't been tampered with.
    void recycle(void *Ptr) {
      Chunk::UnpackedHeader Header;
      Chunk::loadHeader(Allocator.Cookie, Ptr, &Header);
      if (UNLIKELY(Header.State != Chunk::State::Quarantined))
        reportInvalidChunkState(AllocatorAction::Recycling, Ptr);

      Chunk::UnpackedHeader NewHeader = Header;
      NewHeader.State = Chunk::State::Available;
      Chunk::compareExchangeHeader(Allocator.Cookie, Ptr, &NewHeader, &Header);

      void *BlockBegin = Allocator::getBlockBegin(Ptr, &NewHeader);
      const uptr ClassId = NewHeader.ClassId;
      if (LIKELY(ClassId))
        Cache.deallocate(ClassId, BlockBegin);
      else
        Allocator.Secondary.deallocate(BlockBegin);
    }

    // We take a shortcut when allocating a quarantine batch by working with the
    // appropriate class ID instead of using Size. The compiler should optimize
    // the class ID computation and work with the associated cache directly.
    void *allocate(UNUSED uptr Size) {
      const uptr QuarantineClassId = SizeClassMap::getClassIdBySize(
          sizeof(QuarantineBatch) + Chunk::getHeaderSize());
      void *Ptr = Cache.allocate(QuarantineClassId);
      // Quarantine batch allocation failure is fatal.
      if (UNLIKELY(!Ptr))
        reportOutOfMemory(SizeClassMap::getSizeByClassId(QuarantineClassId));

      Ptr = reinterpret_cast<void *>(reinterpret_cast<uptr>(Ptr) +
                                     Chunk::getHeaderSize());
      Chunk::UnpackedHeader Header = {};
      Header.ClassId = QuarantineClassId & Chunk::ClassIdMask;
      Header.SizeOrUnusedBytes = sizeof(QuarantineBatch);
      Header.State = Chunk::State::Allocated;
      Chunk::storeHeader(Allocator.Cookie, Ptr, &Header);

      return Ptr;
    }

    void deallocate(void *Ptr) {
      const uptr QuarantineClassId = SizeClassMap::getClassIdBySize(
          sizeof(QuarantineBatch) + Chunk::getHeaderSize());
      Chunk::UnpackedHeader Header;
      Chunk::loadHeader(Allocator.Cookie, Ptr, &Header);

      if (UNLIKELY(Header.State != Chunk::State::Allocated))
        reportInvalidChunkState(AllocatorAction::Deallocating, Ptr);
      DCHECK_EQ(Header.ClassId, QuarantineClassId);
      DCHECK_EQ(Header.Offset, 0);
      DCHECK_EQ(Header.SizeOrUnusedBytes, sizeof(QuarantineBatch));

      Chunk::UnpackedHeader NewHeader = Header;
      NewHeader.State = Chunk::State::Available;
      Chunk::compareExchangeHeader(Allocator.Cookie, Ptr, &NewHeader, &Header);
      Cache.deallocate(QuarantineClassId,
                       reinterpret_cast<void *>(reinterpret_cast<uptr>(Ptr) -
                                                Chunk::getHeaderSize()));
    }

  private:
    ThisT &Allocator;
    CacheT &Cache;
  };

  typedef GlobalQuarantine<QuarantineCallback, void> QuarantineT;
  typedef typename QuarantineT::CacheT QuarantineCacheT;

  void initLinkerInitialized() {
    performSanityChecks();

    // Check if hardware CRC32 is supported in the binary and by the platform,
    // if so, opt for the CRC32 hardware version of the checksum.
    if (&computeHardwareCRC32 && hasHardwareCRC32())
      HashAlgorithm = Checksum::HardwareCRC32;

    if (UNLIKELY(!getRandom(&Cookie, sizeof(Cookie))))
      Cookie = static_cast<u32>(getMonotonicTime() ^
                                (reinterpret_cast<uptr>(this) >> 4));

    initFlags();
    reportUnrecognizedFlags();

    // Store some flags locally.
    Options.MayReturnNull = getFlags()->may_return_null;
    Options.ZeroContents = getFlags()->zero_contents;
    Options.DeallocTypeMismatch = getFlags()->dealloc_type_mismatch;
    Options.DeleteSizeMismatch = getFlags()->delete_size_mismatch;
    Options.QuarantineMaxChunkSize =
        static_cast<u32>(getFlags()->quarantine_max_chunk_size);

    Stats.initLinkerInitialized();
    Primary.initLinkerInitialized(getFlags()->release_to_os_interval_ms);
    Secondary.initLinkerInitialized(&Stats);

    Quarantine.init(
        static_cast<uptr>(getFlags()->quarantine_size_kb << 10),
        static_cast<uptr>(getFlags()->thread_local_quarantine_size_kb << 10));
  }

  void reset() { memset(this, 0, sizeof(*this)); }

  void unmapTestOnly() {
    TSDRegistry.unmapTestOnly();
    Primary.unmapTestOnly();
  }

  TSDRegistryT *getTSDRegistry() { return &TSDRegistry; }

  // The Cache must be provided zero-initialized.
  void initCache(CacheT *Cache) {
    Cache->initLinkerInitialized(&Stats, &Primary);
  }

  // Release the resources used by a TSD, which involves:
  // - draining the local quarantine cache to the global quarantine;
  // - releasing the cached pointers back to the Primary;
  // - unlinking the local stats from the global ones (destroying the cache does
  //   the last two items).
  void commitBack(TSD<ThisT> *TSD) {
    Quarantine.drain(&TSD->QuarantineCache,
                     QuarantineCallback(*this, TSD->Cache));
    TSD->Cache.destroy(&Stats);
  }

  NOINLINE void *allocate(uptr Size, Chunk::Origin Origin,
                          uptr Alignment = MinAlignment,
                          bool ZeroContents = false) {
    initThreadMaybe();
    ZeroContents |= static_cast<bool>(Options.ZeroContents);

    if (UNLIKELY(Alignment > MaxAlignment)) {
      if (Options.MayReturnNull)
        return nullptr;
      reportAlignmentTooBig(Alignment, MaxAlignment);
    }
    if (Alignment < MinAlignment)
      Alignment = MinAlignment;

    // If the requested size happens to be 0 (more common than you might think),
    // allocate MinAlignment bytes on top of the header. Then add the extra
    // bytes required to fulfill the alignment requirements: we allocate enough
    // to be sure that there will be an address in the block that will satisfy
    // the alignment.
    const uptr NeededSize =
        roundUpTo(Size, MinAlignment) +
        ((Alignment > MinAlignment) ? Alignment : Chunk::getHeaderSize());

    // Takes care of extravagantly large sizes as well as integer overflows.
    static_assert(MaxAllowedMallocSize < UINTPTR_MAX - MaxAlignment, "");
    if (UNLIKELY(Size >= MaxAllowedMallocSize)) {
      if (Options.MayReturnNull)
        return nullptr;
      reportAllocationSizeTooBig(Size, NeededSize, MaxAllowedMallocSize);
    }
    DCHECK_LE(Size, NeededSize);

    void *Block;
    uptr ClassId;
    uptr BlockEnd;
    if (LIKELY(PrimaryT::canAllocate(NeededSize))) {
      ClassId = SizeClassMap::getClassIdBySize(NeededSize);
      DCHECK_NE(ClassId, 0U);
      bool UnlockRequired;
      auto *TSD = TSDRegistry.getTSDAndLock(&UnlockRequired);
      Block = TSD->Cache.allocate(ClassId);
      if (UnlockRequired)
        TSD->unlock();
    } else {
      ClassId = 0;
      Block =
          Secondary.allocate(NeededSize, Alignment, &BlockEnd, ZeroContents);
    }

    if (UNLIKELY(!Block)) {
      if (Options.MayReturnNull)
        return nullptr;
      reportOutOfMemory(NeededSize);
    }

    // We only need to zero the contents for Primary backed allocations. This
    // condition is not necessarily unlikely, but since memset is costly, we
    // might as well mark it as such.
    if (UNLIKELY(ZeroContents && ClassId))
      memset(Block, 0, PrimaryT::getSizeByClassId(ClassId));

    Chunk::UnpackedHeader Header = {};
    uptr UserPtr = reinterpret_cast<uptr>(Block) + Chunk::getHeaderSize();
    if (UNLIKELY(!isAligned(UserPtr, Alignment))) {
      const uptr AlignedUserPtr = roundUpTo(UserPtr, Alignment);
      const uptr Offset = AlignedUserPtr - UserPtr;
      DCHECK_GE(Offset, 2 * sizeof(u32));
      // The BlockMarker has no security purpose, but is specifically meant for
      // the chunk iteration function that can be used in debugging situations.
      // It is the only situation where we have to locate the start of a chunk
      // based on its block address.
      reinterpret_cast<u32 *>(Block)[0] = BlockMarker;
      reinterpret_cast<u32 *>(Block)[1] = static_cast<u32>(Offset);
      UserPtr = AlignedUserPtr;
      Header.Offset = (Offset >> MinAlignmentLog) & Chunk::OffsetMask;
    }
    Header.ClassId = ClassId & Chunk::ClassIdMask;
    Header.State = Chunk::State::Allocated;
    Header.Origin = Origin & Chunk::OriginMask;
    Header.SizeOrUnusedBytes = (ClassId ? Size : BlockEnd - (UserPtr + Size)) &
                               Chunk::SizeOrUnusedBytesMask;
    void *Ptr = reinterpret_cast<void *>(UserPtr);
    Chunk::storeHeader(Cookie, Ptr, &Header);

    if (&__scudo_allocate_hook)
      __scudo_allocate_hook(Ptr, Size);

    return Ptr;
  }

  NOINLINE void deallocate(void *Ptr, Chunk::Origin Origin, uptr DeleteSize = 0,
                           UNUSED uptr Alignment = MinAlignment) {
    // For a deallocation, we only ensure minimal initialization, meaning thread
    // local data will be left uninitialized for now (when using ELF TLS). The
    // fallback cache will be used instead. This is a workaround for a situation
    // where the only heap operation performed in a thread would be a free past
    // the TLS destructors, ending up in initialized thread specific data never
    // being destroyed properly. Any other heap operation will do a full init.
    initThreadMaybe(/*MinimalInit=*/true);

    if (&__scudo_deallocate_hook)
      __scudo_deallocate_hook(Ptr);

    if (UNLIKELY(!Ptr))
      return;
    if (UNLIKELY(!isAligned(reinterpret_cast<uptr>(Ptr), MinAlignment)))
      reportMisalignedPointer(AllocatorAction::Deallocating, Ptr);

    Chunk::UnpackedHeader Header;
    Chunk::loadHeader(Cookie, Ptr, &Header);

    if (UNLIKELY(Header.State != Chunk::State::Allocated))
      reportInvalidChunkState(AllocatorAction::Deallocating, Ptr);
    if (Options.DeallocTypeMismatch) {
      if (Header.Origin != Origin) {
        // With the exception of memalign'd chunks, that can be still be free'd.
        if (UNLIKELY(Header.Origin != Chunk::Origin::Memalign ||
                     Origin != Chunk::Origin::Malloc))
          reportDeallocTypeMismatch(AllocatorAction::Deallocating, Ptr,
                                    Header.Origin, Origin);
      }
    }

    const uptr Size = getSize(Ptr, &Header);
    if (DeleteSize && Options.DeleteSizeMismatch) {
      if (UNLIKELY(DeleteSize != Size))
        reportDeleteSizeMismatch(Ptr, DeleteSize, Size);
    }

    quarantineOrDeallocateChunk(Ptr, &Header, Size);
  }

  void *reallocate(void *OldPtr, uptr NewSize, uptr Alignment = MinAlignment) {
    initThreadMaybe();

    // The following cases are handled by the C wrappers.
    DCHECK_NE(OldPtr, nullptr);
    DCHECK_NE(NewSize, 0);

    if (UNLIKELY(!isAligned(reinterpret_cast<uptr>(OldPtr), MinAlignment)))
      reportMisalignedPointer(AllocatorAction::Reallocating, OldPtr);

    Chunk::UnpackedHeader OldHeader;
    Chunk::loadHeader(Cookie, OldPtr, &OldHeader);

    if (UNLIKELY(OldHeader.State != Chunk::State::Allocated))
      reportInvalidChunkState(AllocatorAction::Reallocating, OldPtr);

    // Pointer has to be allocated with a malloc-type function. Some
    // applications think that it is OK to realloc a memalign'ed pointer, which
    // will trigger this check. It really isn't.
    if (Options.DeallocTypeMismatch) {
      if (UNLIKELY(OldHeader.Origin != Chunk::Origin::Malloc))
        reportDeallocTypeMismatch(AllocatorAction::Reallocating, OldPtr,
                                  OldHeader.Origin, Chunk::Origin::Malloc);
    }

    void *BlockBegin = getBlockBegin(OldPtr, &OldHeader);
    uptr BlockEnd;
    uptr OldSize;
    const uptr ClassId = OldHeader.ClassId;
    if (LIKELY(ClassId)) {
      BlockEnd = reinterpret_cast<uptr>(BlockBegin) +
                 SizeClassMap::getSizeByClassId(ClassId);
      OldSize = OldHeader.SizeOrUnusedBytes;
    } else {
      BlockEnd = SecondaryT::getBlockEnd(BlockBegin);
      OldSize = BlockEnd -
                (reinterpret_cast<uptr>(OldPtr) + OldHeader.SizeOrUnusedBytes);
    }
    // If the new chunk still fits in the previously allocated block (with a
    // reasonable delta), we just keep the old block, and update the chunk
    // header to reflect the size change.
    if (reinterpret_cast<uptr>(OldPtr) + NewSize <= BlockEnd) {
      const uptr Delta =
          OldSize < NewSize ? NewSize - OldSize : OldSize - NewSize;
      if (Delta <= SizeClassMap::MaxSize / 2) {
        Chunk::UnpackedHeader NewHeader = OldHeader;
        NewHeader.SizeOrUnusedBytes =
            (ClassId ? NewSize
                     : BlockEnd - (reinterpret_cast<uptr>(OldPtr) + NewSize)) &
            Chunk::SizeOrUnusedBytesMask;
        Chunk::compareExchangeHeader(Cookie, OldPtr, &NewHeader, &OldHeader);
        return OldPtr;
      }
    }

    // Otherwise we allocate a new one, and deallocate the old one. Some
    // allocators will allocate an even larger chunk (by a fixed factor) to
    // allow for potential further in-place realloc. The gains of such a trick
    // are currently unclear.
    void *NewPtr = allocate(NewSize, Chunk::Origin::Malloc, Alignment);
    if (NewPtr) {
      const uptr OldSize = getSize(OldPtr, &OldHeader);
      memcpy(NewPtr, OldPtr, Min(NewSize, OldSize));
      quarantineOrDeallocateChunk(OldPtr, &OldHeader, OldSize);
    }
    return NewPtr;
  }

  // TODO(kostyak): while this locks the Primary & Secondary, it still allows
  //                pointers to be fetched from the TSD. We ultimately want to
  //                lock the registry as well. For now, it's good enough.
  void disable() {
    initThreadMaybe();
    Primary.disable();
    Secondary.disable();
  }

  void enable() {
    initThreadMaybe();
    Secondary.enable();
    Primary.enable();
  }

  // The function returns the amount of bytes required to store the statistics,
  // which might be larger than the amount of bytes provided. Note that the
  // statistics buffer is not necessarily constant between calls to this
  // function. This can be called with a null buffer or zero size for buffer
  // sizing purposes.
  uptr getStats(char *Buffer, uptr Size) {
    ScopedString Str(1024);
    disable();
    const uptr Length = getStats(&Str) + 1;
    enable();
    if (Length < Size)
      Size = Length;
    if (Buffer && Size) {
      memcpy(Buffer, Str.data(), Size);
      Buffer[Size - 1] = '\0';
    }
    return Length;
  }

  void printStats() {
    ScopedString Str(1024);
    disable();
    getStats(&Str);
    enable();
    Str.output();
  }

  void releaseToOS() {
    initThreadMaybe();
    Primary.releaseToOS();
  }

  // Iterate over all chunks and call a callback for all busy chunks located
  // within the provided memory range. Said callback must not use this allocator
  // or a deadlock can ensue. This fits Android's malloc_iterate() needs.
  void iterateOverChunks(uptr Base, uptr Size, iterate_callback Callback,
                         void *Arg) {
    initThreadMaybe();
    const uptr From = Base;
    const uptr To = Base + Size;
    auto Lambda = [this, From, To, Callback, Arg](uptr Block) {
      if (Block < From || Block >= To)
        return;
      uptr Chunk;
      Chunk::UnpackedHeader Header;
      if (getChunkFromBlock(Block, &Chunk, &Header) &&
          Header.State == Chunk::State::Allocated)
        Callback(Chunk, getSize(reinterpret_cast<void *>(Chunk), &Header), Arg);
    };
    Primary.iterateOverBlocks(Lambda);
    Secondary.iterateOverBlocks(Lambda);
  }

  bool canReturnNull() {
    initThreadMaybe();
    return Options.MayReturnNull;
  }

  // TODO(kostyak): implement this as a "backend" to mallopt.
  bool setOption(UNUSED uptr Option, UNUSED uptr Value) { return false; }

  // Return the usable size for a given chunk. Technically we lie, as we just
  // report the actual size of a chunk. This is done to counteract code actively
  // writing past the end of a chunk (like sqlite3) when the usable size allows
  // for it, which then forces realloc to copy the usable size of a chunk as
  // opposed to its actual size.
  uptr getUsableSize(const void *Ptr) {
    initThreadMaybe();
    if (UNLIKELY(!Ptr))
      return 0;
    Chunk::UnpackedHeader Header;
    Chunk::loadHeader(Cookie, Ptr, &Header);
    // Getting the usable size of a chunk only makes sense if it's allocated.
    if (UNLIKELY(Header.State != Chunk::State::Allocated))
      reportInvalidChunkState(AllocatorAction::Sizing, const_cast<void *>(Ptr));
    return getSize(Ptr, &Header);
  }

  void getStats(StatCounters S) {
    initThreadMaybe();
    Stats.get(S);
  }

  // Returns true if the pointer provided was allocated by the current
  // allocator instance, which is compliant with tcmalloc's ownership concept.
  // A corrupted chunk will not be reported as owned, which is WAI.
  bool isOwned(const void *Ptr) {
    initThreadMaybe();
    if (!Ptr || !isAligned(reinterpret_cast<uptr>(Ptr), MinAlignment))
      return false;
    Chunk::UnpackedHeader Header;
    return Chunk::isValid(Cookie, Ptr, &Header) &&
           Header.State == Chunk::State::Allocated;
  }

private:
  using SecondaryT = typename Params::Secondary;
  typedef typename PrimaryT::SizeClassMap SizeClassMap;

  static const uptr MinAlignmentLog = SCUDO_MIN_ALIGNMENT_LOG;
  static const uptr MaxAlignmentLog = 24U; // 16 MB seems reasonable.
  static const uptr MinAlignment = 1UL << MinAlignmentLog;
  static const uptr MaxAlignment = 1UL << MaxAlignmentLog;
  static const uptr MaxAllowedMallocSize =
      FIRST_32_SECOND_64(1UL << 31, 1ULL << 40);

  static_assert(MinAlignment >= sizeof(Chunk::PackedHeader),
                "Minimal alignment must at least cover a chunk header.");

  static const u32 BlockMarker = 0x44554353U;

  GlobalStats Stats;
  TSDRegistryT TSDRegistry;
  PrimaryT Primary;
  SecondaryT Secondary;
  QuarantineT Quarantine;

  u32 Cookie;

  struct {
    u8 MayReturnNull : 1;       // may_return_null
    u8 ZeroContents : 1;        // zero_contents
    u8 DeallocTypeMismatch : 1; // dealloc_type_mismatch
    u8 DeleteSizeMismatch : 1;  // delete_size_mismatch
    u32 QuarantineMaxChunkSize; // quarantine_max_chunk_size
  } Options;

  // The following might get optimized out by the compiler.
  NOINLINE void performSanityChecks() {
    // Verify that the header offset field can hold the maximum offset. In the
    // case of the Secondary allocator, it takes care of alignment and the
    // offset will always be small. In the case of the Primary, the worst case
    // scenario happens in the last size class, when the backend allocation
    // would already be aligned on the requested alignment, which would happen
    // to be the maximum alignment that would fit in that size class. As a
    // result, the maximum offset will be at most the maximum alignment for the
    // last size class minus the header size, in multiples of MinAlignment.
    Chunk::UnpackedHeader Header = {};
    const uptr MaxPrimaryAlignment = 1UL << getMostSignificantSetBitIndex(
                                         SizeClassMap::MaxSize - MinAlignment);
    const uptr MaxOffset =
        (MaxPrimaryAlignment - Chunk::getHeaderSize()) >> MinAlignmentLog;
    Header.Offset = MaxOffset & Chunk::OffsetMask;
    if (UNLIKELY(Header.Offset != MaxOffset))
      reportSanityCheckError("offset");

    // Verify that we can fit the maximum size or amount of unused bytes in the
    // header. Given that the Secondary fits the allocation to a page, the worst
    // case scenario happens in the Primary. It will depend on the second to
    // last and last class sizes, as well as the dynamic base for the Primary.
    // The following is an over-approximation that works for our needs.
    const uptr MaxSizeOrUnusedBytes = SizeClassMap::MaxSize - 1;
    Header.SizeOrUnusedBytes = MaxSizeOrUnusedBytes;
    if (UNLIKELY(Header.SizeOrUnusedBytes != MaxSizeOrUnusedBytes))
      reportSanityCheckError("size (or unused bytes)");

    const uptr LargestClassId = SizeClassMap::LargestClassId;
    Header.ClassId = LargestClassId;
    if (UNLIKELY(Header.ClassId != LargestClassId))
      reportSanityCheckError("class ID");
  }

  static inline void *getBlockBegin(const void *Ptr,
                                    Chunk::UnpackedHeader *Header) {
    return reinterpret_cast<void *>(
        reinterpret_cast<uptr>(Ptr) - Chunk::getHeaderSize() -
        (static_cast<uptr>(Header->Offset) << MinAlignmentLog));
  }

  // Return the size of a chunk as requested during its allocation.
  inline uptr getSize(const void *Ptr, Chunk::UnpackedHeader *Header) {
    const uptr SizeOrUnusedBytes = Header->SizeOrUnusedBytes;
    if (LIKELY(Header->ClassId))
      return SizeOrUnusedBytes;
    return SecondaryT::getBlockEnd(getBlockBegin(Ptr, Header)) -
           reinterpret_cast<uptr>(Ptr) - SizeOrUnusedBytes;
  }

  ALWAYS_INLINE void initThreadMaybe(bool MinimalInit = false) {
    TSDRegistry.initThreadMaybe(this, MinimalInit);
  }

  void quarantineOrDeallocateChunk(void *Ptr, Chunk::UnpackedHeader *Header,
                                   uptr Size) {
    Chunk::UnpackedHeader NewHeader = *Header;
    // If the quarantine is disabled, the actual size of a chunk is 0 or larger
    // than the maximum allowed, we return a chunk directly to the backend.
    // Logical Or can be short-circuited, which introduces unnecessary
    // conditional jumps, so use bitwise Or and let the compiler be clever.
    const bool BypassQuarantine = !Quarantine.getCacheSize() | !Size |
                                  (Size > Options.QuarantineMaxChunkSize);
    if (BypassQuarantine) {
      NewHeader.State = Chunk::State::Available;
      Chunk::compareExchangeHeader(Cookie, Ptr, &NewHeader, Header);
      void *BlockBegin = getBlockBegin(Ptr, &NewHeader);
      const uptr ClassId = NewHeader.ClassId;
      if (LIKELY(ClassId)) {
        bool UnlockRequired;
        auto *TSD = TSDRegistry.getTSDAndLock(&UnlockRequired);
        TSD->Cache.deallocate(ClassId, BlockBegin);
        if (UnlockRequired)
          TSD->unlock();
      } else {
        Secondary.deallocate(BlockBegin);
      }
    } else {
      NewHeader.State = Chunk::State::Quarantined;
      Chunk::compareExchangeHeader(Cookie, Ptr, &NewHeader, Header);
      bool UnlockRequired;
      auto *TSD = TSDRegistry.getTSDAndLock(&UnlockRequired);
      Quarantine.put(&TSD->QuarantineCache,
                     QuarantineCallback(*this, TSD->Cache), Ptr, Size);
      if (UnlockRequired)
        TSD->unlock();
    }
  }

  bool getChunkFromBlock(uptr Block, uptr *Chunk,
                         Chunk::UnpackedHeader *Header) {
    u32 Offset = 0;
    if (reinterpret_cast<u32 *>(Block)[0] == BlockMarker)
      Offset = reinterpret_cast<u32 *>(Block)[1];
    *Chunk = Block + Offset + Chunk::getHeaderSize();
    return Chunk::isValid(Cookie, reinterpret_cast<void *>(*Chunk), Header);
  }

  uptr getStats(ScopedString *Str) {
    Primary.getStats(Str);
    Secondary.getStats(Str);
    Quarantine.getStats(Str);
    return Str->length();
  }
};

} // namespace scudo

#endif // SCUDO_COMBINED_H_
