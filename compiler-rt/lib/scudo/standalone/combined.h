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
#include "local_cache.h"
#include "memtag.h"
#include "quarantine.h"
#include "report.h"
#include "secondary.h"
#include "stack_depot.h"
#include "string_utils.h"
#include "tsd.h"

#include "scudo/interface.h"

#ifdef GWP_ASAN_HOOKS
#include "gwp_asan/guarded_pool_allocator.h"
#include "gwp_asan/optional/backtrace.h"
#include "gwp_asan/optional/segv_handler.h"
#endif // GWP_ASAN_HOOKS

extern "C" inline void EmptyCallback() {}

#ifdef HAVE_ANDROID_UNSAFE_FRAME_POINTER_CHASE
// This function is not part of the NDK so it does not appear in any public
// header files. We only declare/use it when targeting the platform.
extern "C" size_t android_unsafe_frame_pointer_chase(scudo::uptr *buf,
                                                     size_t num_entries);
#endif

namespace scudo {

template <class Params, void (*PostInitCallback)(void) = EmptyCallback>
class Allocator {
public:
  using PrimaryT = typename Params::Primary;
  using CacheT = typename PrimaryT::CacheT;
  typedef Allocator<Params, PostInitCallback> ThisT;
  typedef typename Params::template TSDRegistryT<ThisT> TSDRegistryT;

  void callPostInitCallback() {
    static pthread_once_t OnceControl = PTHREAD_ONCE_INIT;
    pthread_once(&OnceControl, PostInitCallback);
  }

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
    Options.FillContents =
        getFlags()->zero_contents
            ? ZeroFill
            : (getFlags()->pattern_fill_contents ? PatternOrZeroFill : NoFill);
    Options.DeallocTypeMismatch = getFlags()->dealloc_type_mismatch;
    Options.DeleteSizeMismatch = getFlags()->delete_size_mismatch;
    Options.TrackAllocationStacks = false;
    Options.UseOddEvenTags = true;
    Options.QuarantineMaxChunkSize =
        static_cast<u32>(getFlags()->quarantine_max_chunk_size);

    Stats.initLinkerInitialized();
    const s32 ReleaseToOsIntervalMs = getFlags()->release_to_os_interval_ms;
    Primary.initLinkerInitialized(ReleaseToOsIntervalMs);
    Secondary.initLinkerInitialized(&Stats, ReleaseToOsIntervalMs);

    Quarantine.init(
        static_cast<uptr>(getFlags()->quarantine_size_kb << 10),
        static_cast<uptr>(getFlags()->thread_local_quarantine_size_kb << 10));
  }

  // Initialize the embedded GWP-ASan instance. Requires the main allocator to
  // be functional, best called from PostInitCallback.
  void initGwpAsan() {
#ifdef GWP_ASAN_HOOKS
    gwp_asan::options::Options Opt;
    Opt.Enabled = getFlags()->GWP_ASAN_Enabled;
    // Bear in mind - Scudo has its own alignment guarantees that are strictly
    // enforced. Scudo exposes the same allocation function for everything from
    // malloc() to posix_memalign, so in general this flag goes unused, as Scudo
    // will always ask GWP-ASan for an aligned amount of bytes.
    Opt.PerfectlyRightAlign = getFlags()->GWP_ASAN_PerfectlyRightAlign;
    Opt.MaxSimultaneousAllocations =
        getFlags()->GWP_ASAN_MaxSimultaneousAllocations;
    Opt.SampleRate = getFlags()->GWP_ASAN_SampleRate;
    Opt.InstallSignalHandlers = getFlags()->GWP_ASAN_InstallSignalHandlers;
    // Embedded GWP-ASan is locked through the Scudo atfork handler (via
    // Allocator::disable calling GWPASan.disable). Disable GWP-ASan's atfork
    // handler.
    Opt.InstallForkHandlers = false;
    Opt.Backtrace = gwp_asan::options::getBacktraceFunction();
    GuardedAlloc.init(Opt);

    if (Opt.InstallSignalHandlers)
      gwp_asan::crash_handler::installSignalHandlers(
          &GuardedAlloc, Printf, gwp_asan::options::getPrintBacktraceFunction(),
          gwp_asan::crash_handler::getSegvBacktraceFunction());
#endif // GWP_ASAN_HOOKS
  }

  ALWAYS_INLINE void initThreadMaybe(bool MinimalInit = false) {
    TSDRegistry.initThreadMaybe(this, MinimalInit);
  }

  void reset() { memset(this, 0, sizeof(*this)); }

  void unmapTestOnly() {
    TSDRegistry.unmapTestOnly();
    Primary.unmapTestOnly();
#ifdef GWP_ASAN_HOOKS
    if (getFlags()->GWP_ASAN_InstallSignalHandlers)
      gwp_asan::crash_handler::uninstallSignalHandlers();
    GuardedAlloc.uninitTestOnly();
#endif // GWP_ASAN_HOOKS
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

  ALWAYS_INLINE void *untagPointerMaybe(void *Ptr) {
    if (Primary.SupportsMemoryTagging)
      return reinterpret_cast<void *>(
          untagPointer(reinterpret_cast<uptr>(Ptr)));
    return Ptr;
  }

  NOINLINE u32 collectStackTrace() {
#ifdef HAVE_ANDROID_UNSAFE_FRAME_POINTER_CHASE
    // Discard collectStackTrace() frame and allocator function frame.
    constexpr uptr DiscardFrames = 2;
    uptr Stack[MaxTraceSize + DiscardFrames];
    uptr Size =
        android_unsafe_frame_pointer_chase(Stack, MaxTraceSize + DiscardFrames);
    Size = Min<uptr>(Size, MaxTraceSize + DiscardFrames);
    return Depot.insert(Stack + Min<uptr>(DiscardFrames, Size), Stack + Size);
#else
    return 0;
#endif
  }

  uptr computeOddEvenMaskForPointerMaybe(uptr Ptr, uptr Size) {
    if (!Options.UseOddEvenTags)
      return 0;

    // If a chunk's tag is odd, we want the tags of the surrounding blocks to be
    // even, and vice versa. Blocks are laid out Size bytes apart, and adding
    // Size to Ptr will flip the least significant set bit of Size in Ptr, so
    // that bit will have the pattern 010101... for consecutive blocks, which we
    // can use to determine which tag mask to use.
    return (Ptr & (1ULL << getLeastSignificantSetBitIndex(Size))) ? 0xaaaa
                                                                  : 0x5555;
  }

  NOINLINE void *allocate(uptr Size, Chunk::Origin Origin,
                          uptr Alignment = MinAlignment,
                          bool ZeroContents = false) {
    initThreadMaybe();

#ifdef GWP_ASAN_HOOKS
    if (UNLIKELY(GuardedAlloc.shouldSample())) {
      if (void *Ptr = GuardedAlloc.allocate(roundUpTo(Size, Alignment)))
        return Ptr;
    }
#endif // GWP_ASAN_HOOKS

    const FillContentsMode FillContents = ZeroContents ? ZeroFill
                                          : TSDRegistry.getDisableMemInit()
                                              ? NoFill
                                              : Options.FillContents;

    if (UNLIKELY(Alignment > MaxAlignment)) {
      if (Options.MayReturnNull)
        return nullptr;
      reportAlignmentTooBig(Alignment, MaxAlignment);
    }
    if (UNLIKELY(Alignment < MinAlignment))
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

    void *Block = nullptr;
    uptr ClassId = 0;
    uptr SecondaryBlockEnd = 0;
    if (LIKELY(PrimaryT::canAllocate(NeededSize))) {
      ClassId = SizeClassMap::getClassIdBySize(NeededSize);
      DCHECK_NE(ClassId, 0U);
      bool UnlockRequired;
      auto *TSD = TSDRegistry.getTSDAndLock(&UnlockRequired);
      Block = TSD->Cache.allocate(ClassId);
      // If the allocation failed, the most likely reason with a 32-bit primary
      // is the region being full. In that event, retry in each successively
      // larger class until it fits. If it fails to fit in the largest class,
      // fallback to the Secondary.
      if (UNLIKELY(!Block)) {
        while (ClassId < SizeClassMap::LargestClassId) {
          Block = TSD->Cache.allocate(++ClassId);
          if (LIKELY(Block))
            break;
        }
        if (UNLIKELY(!Block))
          ClassId = 0;
      }
      if (UnlockRequired)
        TSD->unlock();
    }
    if (UNLIKELY(ClassId == 0))
      Block = Secondary.allocate(NeededSize, Alignment, &SecondaryBlockEnd,
                                 FillContents);

    if (UNLIKELY(!Block)) {
      if (Options.MayReturnNull)
        return nullptr;
      reportOutOfMemory(NeededSize);
    }

    const uptr BlockUptr = reinterpret_cast<uptr>(Block);
    const uptr UnalignedUserPtr = BlockUptr + Chunk::getHeaderSize();
    const uptr UserPtr = roundUpTo(UnalignedUserPtr, Alignment);

    void *Ptr = reinterpret_cast<void *>(UserPtr);
    void *TaggedPtr = Ptr;
    if (LIKELY(ClassId)) {
      // We only need to zero or tag the contents for Primary backed
      // allocations. We only set tags for primary allocations in order to avoid
      // faulting potentially large numbers of pages for large secondary
      // allocations. We assume that guard pages are enough to protect these
      // allocations.
      //
      // FIXME: When the kernel provides a way to set the background tag of a
      // mapping, we should be able to tag secondary allocations as well.
      //
      // When memory tagging is enabled, zeroing the contents is done as part of
      // setting the tag.
      if (UNLIKELY(useMemoryTagging())) {
        uptr PrevUserPtr;
        Chunk::UnpackedHeader Header;
        const uptr BlockSize = PrimaryT::getSizeByClassId(ClassId);
        const uptr BlockEnd = BlockUptr + BlockSize;
        // If possible, try to reuse the UAF tag that was set by deallocate().
        // For simplicity, only reuse tags if we have the same start address as
        // the previous allocation. This handles the majority of cases since
        // most allocations will not be more aligned than the minimum alignment.
        //
        // We need to handle situations involving reclaimed chunks, and retag
        // the reclaimed portions if necessary. In the case where the chunk is
        // fully reclaimed, the chunk's header will be zero, which will trigger
        // the code path for new mappings and invalid chunks that prepares the
        // chunk from scratch. There are three possibilities for partial
        // reclaiming:
        //
        // (1) Header was reclaimed, data was partially reclaimed.
        // (2) Header was not reclaimed, all data was reclaimed (e.g. because
        //     data started on a page boundary).
        // (3) Header was not reclaimed, data was partially reclaimed.
        //
        // Case (1) will be handled in the same way as for full reclaiming,
        // since the header will be zero.
        //
        // We can detect case (2) by loading the tag from the start
        // of the chunk. If it is zero, it means that either all data was
        // reclaimed (since we never use zero as the chunk tag), or that the
        // previous allocation was of size zero. Either way, we need to prepare
        // a new chunk from scratch.
        //
        // We can detect case (3) by moving to the next page (if covered by the
        // chunk) and loading the tag of its first granule. If it is zero, it
        // means that all following pages may need to be retagged. On the other
        // hand, if it is nonzero, we can assume that all following pages are
        // still tagged, according to the logic that if any of the pages
        // following the next page were reclaimed, the next page would have been
        // reclaimed as well.
        uptr TaggedUserPtr;
        if (getChunkFromBlock(BlockUptr, &PrevUserPtr, &Header) &&
            PrevUserPtr == UserPtr &&
            (TaggedUserPtr = loadTag(UserPtr)) != UserPtr) {
          uptr PrevEnd = TaggedUserPtr + Header.SizeOrUnusedBytes;
          const uptr NextPage = roundUpTo(TaggedUserPtr, getPageSizeCached());
          if (NextPage < PrevEnd && loadTag(NextPage) != NextPage)
            PrevEnd = NextPage;
          TaggedPtr = reinterpret_cast<void *>(TaggedUserPtr);
          resizeTaggedChunk(PrevEnd, TaggedUserPtr + Size, BlockEnd);
          if (UNLIKELY(FillContents != NoFill && !Header.OriginOrWasZeroed)) {
            // If an allocation needs to be zeroed (i.e. calloc) we can normally
            // avoid zeroing the memory now since we can rely on memory having
            // been zeroed on free, as this is normally done while setting the
            // UAF tag. But if tagging was disabled per-thread when the memory
            // was freed, it would not have been retagged and thus zeroed, and
            // therefore it needs to be zeroed now.
            memset(TaggedPtr, 0,
                   Min(Size, roundUpTo(PrevEnd - TaggedUserPtr,
                                       archMemoryTagGranuleSize())));
          } else if (Size) {
            // Clear any stack metadata that may have previously been stored in
            // the chunk data.
            memset(TaggedPtr, 0, archMemoryTagGranuleSize());
          }
        } else {
          const uptr OddEvenMask =
              computeOddEvenMaskForPointerMaybe(BlockUptr, BlockSize);
          TaggedPtr = prepareTaggedChunk(Ptr, Size, OddEvenMask, BlockEnd);
        }
        storeAllocationStackMaybe(Ptr);
      } else if (UNLIKELY(FillContents != NoFill)) {
        // This condition is not necessarily unlikely, but since memset is
        // costly, we might as well mark it as such.
        memset(Block, FillContents == ZeroFill ? 0 : PatternFillByte,
               PrimaryT::getSizeByClassId(ClassId));
      }
    }

    Chunk::UnpackedHeader Header = {};
    if (UNLIKELY(UnalignedUserPtr != UserPtr)) {
      const uptr Offset = UserPtr - UnalignedUserPtr;
      DCHECK_GE(Offset, 2 * sizeof(u32));
      // The BlockMarker has no security purpose, but is specifically meant for
      // the chunk iteration function that can be used in debugging situations.
      // It is the only situation where we have to locate the start of a chunk
      // based on its block address.
      reinterpret_cast<u32 *>(Block)[0] = BlockMarker;
      reinterpret_cast<u32 *>(Block)[1] = static_cast<u32>(Offset);
      Header.Offset = (Offset >> MinAlignmentLog) & Chunk::OffsetMask;
    }
    Header.ClassId = ClassId & Chunk::ClassIdMask;
    Header.State = Chunk::State::Allocated;
    Header.OriginOrWasZeroed = Origin & Chunk::OriginMask;
    Header.SizeOrUnusedBytes =
        (ClassId ? Size : SecondaryBlockEnd - (UserPtr + Size)) &
        Chunk::SizeOrUnusedBytesMask;
    Chunk::storeHeader(Cookie, Ptr, &Header);

    if (&__scudo_allocate_hook)
      __scudo_allocate_hook(TaggedPtr, Size);

    return TaggedPtr;
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

#ifdef GWP_ASAN_HOOKS
    if (UNLIKELY(GuardedAlloc.pointerIsMine(Ptr))) {
      GuardedAlloc.deallocate(Ptr);
      return;
    }
#endif // GWP_ASAN_HOOKS

    if (&__scudo_deallocate_hook)
      __scudo_deallocate_hook(Ptr);

    if (UNLIKELY(!Ptr))
      return;
    if (UNLIKELY(!isAligned(reinterpret_cast<uptr>(Ptr), MinAlignment)))
      reportMisalignedPointer(AllocatorAction::Deallocating, Ptr);

    Ptr = untagPointerMaybe(Ptr);

    Chunk::UnpackedHeader Header;
    Chunk::loadHeader(Cookie, Ptr, &Header);

    if (UNLIKELY(Header.State != Chunk::State::Allocated))
      reportInvalidChunkState(AllocatorAction::Deallocating, Ptr);
    if (Options.DeallocTypeMismatch) {
      if (Header.OriginOrWasZeroed != Origin) {
        // With the exception of memalign'd chunks, that can be still be free'd.
        if (UNLIKELY(Header.OriginOrWasZeroed != Chunk::Origin::Memalign ||
                     Origin != Chunk::Origin::Malloc))
          reportDeallocTypeMismatch(AllocatorAction::Deallocating, Ptr,
                                    Header.OriginOrWasZeroed, Origin);
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

    if (UNLIKELY(NewSize >= MaxAllowedMallocSize)) {
      if (Options.MayReturnNull)
        return nullptr;
      reportAllocationSizeTooBig(NewSize, 0, MaxAllowedMallocSize);
    }

    void *OldTaggedPtr = OldPtr;
    OldPtr = untagPointerMaybe(OldPtr);

    // The following cases are handled by the C wrappers.
    DCHECK_NE(OldPtr, nullptr);
    DCHECK_NE(NewSize, 0);

#ifdef GWP_ASAN_HOOKS
    if (UNLIKELY(GuardedAlloc.pointerIsMine(OldPtr))) {
      uptr OldSize = GuardedAlloc.getSize(OldPtr);
      void *NewPtr = allocate(NewSize, Chunk::Origin::Malloc, Alignment);
      if (NewPtr)
        memcpy(NewPtr, OldPtr, (NewSize < OldSize) ? NewSize : OldSize);
      GuardedAlloc.deallocate(OldPtr);
      return NewPtr;
    }
#endif // GWP_ASAN_HOOKS

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
      if (UNLIKELY(OldHeader.OriginOrWasZeroed != Chunk::Origin::Malloc))
        reportDeallocTypeMismatch(AllocatorAction::Reallocating, OldPtr,
                                  OldHeader.OriginOrWasZeroed,
                                  Chunk::Origin::Malloc);
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
      if (NewSize > OldSize || (OldSize - NewSize) < getPageSizeCached()) {
        Chunk::UnpackedHeader NewHeader = OldHeader;
        NewHeader.SizeOrUnusedBytes =
            (ClassId ? NewSize
                     : BlockEnd - (reinterpret_cast<uptr>(OldPtr) + NewSize)) &
            Chunk::SizeOrUnusedBytesMask;
        Chunk::compareExchangeHeader(Cookie, OldPtr, &NewHeader, &OldHeader);
        if (UNLIKELY(ClassId && useMemoryTagging())) {
          resizeTaggedChunk(reinterpret_cast<uptr>(OldTaggedPtr) + OldSize,
                            reinterpret_cast<uptr>(OldTaggedPtr) + NewSize,
                            BlockEnd);
          storeAllocationStackMaybe(OldPtr);
        }
        return OldTaggedPtr;
      }
    }

    // Otherwise we allocate a new one, and deallocate the old one. Some
    // allocators will allocate an even larger chunk (by a fixed factor) to
    // allow for potential further in-place realloc. The gains of such a trick
    // are currently unclear.
    void *NewPtr = allocate(NewSize, Chunk::Origin::Malloc, Alignment);
    if (NewPtr) {
      const uptr OldSize = getSize(OldPtr, &OldHeader);
      memcpy(NewPtr, OldTaggedPtr, Min(NewSize, OldSize));
      quarantineOrDeallocateChunk(OldPtr, &OldHeader, OldSize);
    }
    return NewPtr;
  }

  // TODO(kostyak): disable() is currently best-effort. There are some small
  //                windows of time when an allocation could still succeed after
  //                this function finishes. We will revisit that later.
  void disable() {
    initThreadMaybe();
#ifdef GWP_ASAN_HOOKS
    GuardedAlloc.disable();
#endif
    TSDRegistry.disable();
    Stats.disable();
    Quarantine.disable();
    Primary.disable();
    Secondary.disable();
  }

  void enable() {
    initThreadMaybe();
    Secondary.enable();
    Primary.enable();
    Quarantine.enable();
    Stats.enable();
    TSDRegistry.enable();
#ifdef GWP_ASAN_HOOKS
    GuardedAlloc.enable();
#endif
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
    Secondary.releaseToOS();
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
          Header.State == Chunk::State::Allocated) {
        uptr TaggedChunk = Chunk;
        if (useMemoryTagging())
          TaggedChunk = loadTag(Chunk);
        Callback(TaggedChunk, getSize(reinterpret_cast<void *>(Chunk), &Header),
                 Arg);
      }
    };
    Primary.iterateOverBlocks(Lambda);
    Secondary.iterateOverBlocks(Lambda);
#ifdef GWP_ASAN_HOOKS
    GuardedAlloc.iterate(reinterpret_cast<void *>(Base), Size, Callback, Arg);
#endif
  }

  bool canReturnNull() {
    initThreadMaybe();
    return Options.MayReturnNull;
  }

  bool setOption(Option O, sptr Value) {
    initThreadMaybe();
    if (O == Option::MemtagTuning) {
      // Enabling odd/even tags involves a tradeoff between use-after-free
      // detection and buffer overflow detection. Odd/even tags make it more
      // likely for buffer overflows to be detected by increasing the size of
      // the guaranteed "red zone" around the allocation, but on the other hand
      // use-after-free is less likely to be detected because the tag space for
      // any particular chunk is cut in half. Therefore we use this tuning
      // setting to control whether odd/even tags are enabled.
      if (Value == M_MEMTAG_TUNING_BUFFER_OVERFLOW)
        Options.UseOddEvenTags = true;
      else if (Value == M_MEMTAG_TUNING_UAF)
        Options.UseOddEvenTags = false;
      return true;
    } else {
      // We leave it to the various sub-components to decide whether or not they
      // want to handle the option, but we do not want to short-circuit
      // execution if one of the setOption was to return false.
      const bool PrimaryResult = Primary.setOption(O, Value);
      const bool SecondaryResult = Secondary.setOption(O, Value);
      const bool RegistryResult = TSDRegistry.setOption(O, Value);
      return PrimaryResult && SecondaryResult && RegistryResult;
    }
    return false;
  }

  // Return the usable size for a given chunk. Technically we lie, as we just
  // report the actual size of a chunk. This is done to counteract code actively
  // writing past the end of a chunk (like sqlite3) when the usable size allows
  // for it, which then forces realloc to copy the usable size of a chunk as
  // opposed to its actual size.
  uptr getUsableSize(const void *Ptr) {
    initThreadMaybe();
    if (UNLIKELY(!Ptr))
      return 0;

#ifdef GWP_ASAN_HOOKS
    if (UNLIKELY(GuardedAlloc.pointerIsMine(Ptr)))
      return GuardedAlloc.getSize(Ptr);
#endif // GWP_ASAN_HOOKS

    Ptr = untagPointerMaybe(const_cast<void *>(Ptr));
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
#ifdef GWP_ASAN_HOOKS
    if (GuardedAlloc.pointerIsMine(Ptr))
      return true;
#endif // GWP_ASAN_HOOKS
    if (!Ptr || !isAligned(reinterpret_cast<uptr>(Ptr), MinAlignment))
      return false;
    Ptr = untagPointerMaybe(const_cast<void *>(Ptr));
    Chunk::UnpackedHeader Header;
    return Chunk::isValid(Cookie, Ptr, &Header) &&
           Header.State == Chunk::State::Allocated;
  }

  bool useMemoryTagging() { return Primary.useMemoryTagging(); }

  void disableMemoryTagging() { Primary.disableMemoryTagging(); }

  void setTrackAllocationStacks(bool Track) {
    initThreadMaybe();
    Options.TrackAllocationStacks = Track;
  }

  void setFillContents(FillContentsMode FillContents) {
    initThreadMaybe();
    Options.FillContents = FillContents;
  }

  const char *getStackDepotAddress() const {
    return reinterpret_cast<const char *>(&Depot);
  }

  const char *getRegionInfoArrayAddress() const {
    return Primary.getRegionInfoArrayAddress();
  }

  static uptr getRegionInfoArraySize() {
    return PrimaryT::getRegionInfoArraySize();
  }

  static void getErrorInfo(struct scudo_error_info *ErrorInfo,
                           uintptr_t FaultAddr, const char *DepotPtr,
                           const char *RegionInfoPtr, const char *Memory,
                           const char *MemoryTags, uintptr_t MemoryAddr,
                           size_t MemorySize) {
    *ErrorInfo = {};
    if (!PrimaryT::SupportsMemoryTagging ||
        MemoryAddr + MemorySize < MemoryAddr)
      return;

    uptr UntaggedFaultAddr = untagPointer(FaultAddr);
    u8 FaultAddrTag = extractTag(FaultAddr);
    BlockInfo Info =
        PrimaryT::findNearestBlock(RegionInfoPtr, UntaggedFaultAddr);

    auto GetGranule = [&](uptr Addr, const char **Data, uint8_t *Tag) -> bool {
      if (Addr < MemoryAddr || Addr + archMemoryTagGranuleSize() < Addr ||
          Addr + archMemoryTagGranuleSize() > MemoryAddr + MemorySize)
        return false;
      *Data = &Memory[Addr - MemoryAddr];
      *Tag = static_cast<u8>(
          MemoryTags[(Addr - MemoryAddr) / archMemoryTagGranuleSize()]);
      return true;
    };

    auto ReadBlock = [&](uptr Addr, uptr *ChunkAddr,
                         Chunk::UnpackedHeader *Header, const u32 **Data,
                         u8 *Tag) {
      const char *BlockBegin;
      u8 BlockBeginTag;
      if (!GetGranule(Addr, &BlockBegin, &BlockBeginTag))
        return false;
      uptr ChunkOffset = getChunkOffsetFromBlock(BlockBegin);
      *ChunkAddr = Addr + ChunkOffset;

      const char *ChunkBegin;
      if (!GetGranule(*ChunkAddr, &ChunkBegin, Tag))
        return false;
      *Header = *reinterpret_cast<const Chunk::UnpackedHeader *>(
          ChunkBegin - Chunk::getHeaderSize());
      *Data = reinterpret_cast<const u32 *>(ChunkBegin);
      return true;
    };

    auto *Depot = reinterpret_cast<const StackDepot *>(DepotPtr);

    auto MaybeCollectTrace = [&](uintptr_t(&Trace)[MaxTraceSize], u32 Hash) {
      uptr RingPos, Size;
      if (!Depot->find(Hash, &RingPos, &Size))
        return;
      for (unsigned I = 0; I != Size && I != MaxTraceSize; ++I)
        Trace[I] = (*Depot)[RingPos + I];
    };

    size_t NextErrorReport = 0;

    // First, check for UAF.
    {
      uptr ChunkAddr;
      Chunk::UnpackedHeader Header;
      const u32 *Data;
      uint8_t Tag;
      if (ReadBlock(Info.BlockBegin, &ChunkAddr, &Header, &Data, &Tag) &&
          Header.State != Chunk::State::Allocated &&
          Data[MemTagPrevTagIndex] == FaultAddrTag) {
        auto *R = &ErrorInfo->reports[NextErrorReport++];
        R->error_type = USE_AFTER_FREE;
        R->allocation_address = ChunkAddr;
        R->allocation_size = Header.SizeOrUnusedBytes;
        MaybeCollectTrace(R->allocation_trace,
                          Data[MemTagAllocationTraceIndex]);
        R->allocation_tid = Data[MemTagAllocationTidIndex];
        MaybeCollectTrace(R->deallocation_trace,
                          Data[MemTagDeallocationTraceIndex]);
        R->deallocation_tid = Data[MemTagDeallocationTidIndex];
      }
    }

    auto CheckOOB = [&](uptr BlockAddr) {
      if (BlockAddr < Info.RegionBegin || BlockAddr >= Info.RegionEnd)
        return false;

      uptr ChunkAddr;
      Chunk::UnpackedHeader Header;
      const u32 *Data;
      uint8_t Tag;
      if (!ReadBlock(BlockAddr, &ChunkAddr, &Header, &Data, &Tag) ||
          Header.State != Chunk::State::Allocated || Tag != FaultAddrTag)
        return false;

      auto *R = &ErrorInfo->reports[NextErrorReport++];
      R->error_type =
          UntaggedFaultAddr < ChunkAddr ? BUFFER_UNDERFLOW : BUFFER_OVERFLOW;
      R->allocation_address = ChunkAddr;
      R->allocation_size = Header.SizeOrUnusedBytes;
      MaybeCollectTrace(R->allocation_trace, Data[MemTagAllocationTraceIndex]);
      R->allocation_tid = Data[MemTagAllocationTidIndex];
      return NextErrorReport ==
             sizeof(ErrorInfo->reports) / sizeof(ErrorInfo->reports[0]);
    };

    if (CheckOOB(Info.BlockBegin))
      return;

    // Check for OOB in the 30 surrounding blocks. Beyond that we are likely to
    // hit false positives.
    for (int I = 1; I != 16; ++I)
      if (CheckOOB(Info.BlockBegin + I * Info.BlockSize) ||
          CheckOOB(Info.BlockBegin - I * Info.BlockSize))
        return;
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
  static_assert(!PrimaryT::SupportsMemoryTagging ||
                    MinAlignment >= archMemoryTagGranuleSize(),
                "");

  static const u32 BlockMarker = 0x44554353U;

  // These are indexes into an "array" of 32-bit values that store information
  // inline with a chunk that is relevant to diagnosing memory tag faults, where
  // 0 corresponds to the address of the user memory. This means that negative
  // indexes may be used to store information about allocations, while positive
  // indexes may only be used to store information about deallocations, because
  // the user memory is in use until it has been deallocated. The smallest index
  // that may be used is -2, which corresponds to 8 bytes before the user
  // memory, because the chunk header size is 8 bytes and in allocators that
  // support memory tagging the minimum alignment is at least the tag granule
  // size (16 on aarch64), and the largest index that may be used is 3 because
  // we are only guaranteed to have at least a granule's worth of space in the
  // user memory.
  static const sptr MemTagAllocationTraceIndex = -2;
  static const sptr MemTagAllocationTidIndex = -1;
  static const sptr MemTagDeallocationTraceIndex = 0;
  static const sptr MemTagDeallocationTidIndex = 1;
  static const sptr MemTagPrevTagIndex = 2;

  static const uptr MaxTraceSize = 64;

  GlobalStats Stats;
  TSDRegistryT TSDRegistry;
  PrimaryT Primary;
  SecondaryT Secondary;
  QuarantineT Quarantine;

  u32 Cookie;

  struct {
    u8 MayReturnNull : 1;              // may_return_null
    FillContentsMode FillContents : 2; // zero_contents, pattern_fill_contents
    u8 DeallocTypeMismatch : 1;        // dealloc_type_mismatch
    u8 DeleteSizeMismatch : 1;         // delete_size_mismatch
    u8 TrackAllocationStacks : 1;
    u8 UseOddEvenTags : 1;
    u32 QuarantineMaxChunkSize; // quarantine_max_chunk_size
  } Options;

#ifdef GWP_ASAN_HOOKS
  gwp_asan::GuardedPoolAllocator GuardedAlloc;
#endif // GWP_ASAN_HOOKS

  StackDepot Depot;

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

  void quarantineOrDeallocateChunk(void *Ptr, Chunk::UnpackedHeader *Header,
                                   uptr Size) {
    Chunk::UnpackedHeader NewHeader = *Header;
    if (UNLIKELY(NewHeader.ClassId && useMemoryTagging())) {
      u8 PrevTag = extractTag(loadTag(reinterpret_cast<uptr>(Ptr)));
      if (!TSDRegistry.getDisableMemInit()) {
        uptr TaggedBegin, TaggedEnd;
        const uptr OddEvenMask = computeOddEvenMaskForPointerMaybe(
            reinterpret_cast<uptr>(getBlockBegin(Ptr, &NewHeader)),
            SizeClassMap::getSizeByClassId(NewHeader.ClassId));
        // Exclude the previous tag so that immediate use after free is detected
        // 100% of the time.
        setRandomTag(Ptr, Size, OddEvenMask | (1UL << PrevTag), &TaggedBegin,
                     &TaggedEnd);
      }
      NewHeader.OriginOrWasZeroed = !TSDRegistry.getDisableMemInit();
      storeDeallocationStackMaybe(Ptr, PrevTag);
    }
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
    *Chunk =
        Block + getChunkOffsetFromBlock(reinterpret_cast<const char *>(Block));
    return Chunk::isValid(Cookie, reinterpret_cast<void *>(*Chunk), Header);
  }

  static uptr getChunkOffsetFromBlock(const char *Block) {
    u32 Offset = 0;
    if (reinterpret_cast<const u32 *>(Block)[0] == BlockMarker)
      Offset = reinterpret_cast<const u32 *>(Block)[1];
    return Offset + Chunk::getHeaderSize();
  }

  void storeAllocationStackMaybe(void *Ptr) {
    if (!UNLIKELY(Options.TrackAllocationStacks))
      return;
    auto *Ptr32 = reinterpret_cast<u32 *>(Ptr);
    Ptr32[MemTagAllocationTraceIndex] = collectStackTrace();
    Ptr32[MemTagAllocationTidIndex] = getThreadID();
  }

  void storeDeallocationStackMaybe(void *Ptr, uint8_t PrevTag) {
    if (!UNLIKELY(Options.TrackAllocationStacks))
      return;

    // Disable tag checks here so that we don't need to worry about zero sized
    // allocations.
    ScopedDisableMemoryTagChecks x;
    auto *Ptr32 = reinterpret_cast<u32 *>(Ptr);
    Ptr32[MemTagDeallocationTraceIndex] = collectStackTrace();
    Ptr32[MemTagDeallocationTidIndex] = getThreadID();
    Ptr32[MemTagPrevTagIndex] = PrevTag;
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
