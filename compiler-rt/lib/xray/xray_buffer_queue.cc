//===-- xray_buffer_queue.cc -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of XRay, a dynamic runtime instruementation system.
//
// Defines the interface for a buffer queue implementation.
//
//===----------------------------------------------------------------------===//
#include "xray_buffer_queue.h"
#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_posix.h"
#include "xray_allocator.h"
#include "xray_defs.h"
#include <memory>
#include <sys/mman.h>

using namespace __xray;
using namespace __sanitizer;

namespace {

void decRefCount(unsigned char *ControlBlock, size_t Size, size_t Count) {
  if (ControlBlock == nullptr)
    return;
  auto *RefCount = reinterpret_cast<atomic_uint64_t *>(ControlBlock);
  if (atomic_fetch_sub(RefCount, 1, memory_order_acq_rel) == 1)
    deallocateBuffer(ControlBlock, (Size * Count) + kCacheLineSize);
}

void incRefCount(unsigned char *ControlBlock) {
  if (ControlBlock == nullptr)
    return;
  auto *RefCount = reinterpret_cast<atomic_uint64_t *>(ControlBlock);
  atomic_fetch_add(RefCount, 1, memory_order_acq_rel);
}

} // namespace

BufferQueue::ErrorCode BufferQueue::init(size_t BS, size_t BC) {
  SpinMutexLock Guard(&Mutex);

  if (!finalizing())
    return BufferQueue::ErrorCode::AlreadyInitialized;

  cleanupBuffers();

  bool Success = false;
  BufferSize = BS;
  BufferCount = BC;
  BackingStore = allocateBuffer((BufferSize * BufferCount) + kCacheLineSize);
  if (BackingStore == nullptr)
    return BufferQueue::ErrorCode::NotEnoughMemory;

  auto CleanupBackingStore = __sanitizer::at_scope_exit([&, this] {
    if (Success)
      return;
    deallocateBuffer(BackingStore, (BufferSize * BufferCount) + kCacheLineSize);
    BackingStore = nullptr;
  });

  Buffers = initArray<BufferRep>(BufferCount);
  if (Buffers == nullptr)
    return BufferQueue::ErrorCode::NotEnoughMemory;

  // At this point we increment the generation number to associate the buffers
  // to the new generation.
  atomic_fetch_add(&Generation, 1, memory_order_acq_rel);

  Success = true;

  // First, we initialize the refcount in the RefCountedBackingStore, which we
  // treat as being at the start of the BackingStore pointer.
  auto ControlBlock = reinterpret_cast<atomic_uint64_t *>(BackingStore);
  atomic_store(ControlBlock, 1, memory_order_release);

  for (size_t i = 0; i < BufferCount; ++i) {
    auto &T = Buffers[i];
    auto &Buf = T.Buff;
    atomic_store(&Buf.Extents, 0, memory_order_release);
    Buf.Generation = generation();
    Buf.Data = BackingStore + kCacheLineSize + (BufferSize * i);
    Buf.Size = BufferSize;
    Buf.BackingStore = BackingStore;
    Buf.Count = BufferCount;
    T.Used = false;
  }

  Next = Buffers;
  First = Buffers;
  LiveBuffers = 0;
  atomic_store(&Finalizing, 0, memory_order_release);
  return BufferQueue::ErrorCode::Ok;
}

BufferQueue::BufferQueue(size_t B, size_t N,
                         bool &Success) XRAY_NEVER_INSTRUMENT
    : BufferSize(B),
      BufferCount(N),
      Mutex(),
      Finalizing{1},
      BackingStore(nullptr),
      Buffers(nullptr),
      Next(Buffers),
      First(Buffers),
      LiveBuffers(0),
      Generation{0} {
  Success = init(B, N) == BufferQueue::ErrorCode::Ok;
}

BufferQueue::ErrorCode BufferQueue::getBuffer(Buffer &Buf) {
  if (atomic_load(&Finalizing, memory_order_acquire))
    return ErrorCode::QueueFinalizing;

  BufferRep *B = nullptr;
  {
    SpinMutexLock Guard(&Mutex);
    if (LiveBuffers == BufferCount)
      return ErrorCode::NotEnoughMemory;
    B = Next++;
    if (Next == (Buffers + BufferCount))
      Next = Buffers;
    ++LiveBuffers;
  }

  incRefCount(BackingStore);
  Buf.Data = B->Buff.Data;
  Buf.Generation = generation();
  Buf.Size = B->Buff.Size;
  Buf.BackingStore = BackingStore;
  Buf.Count = BufferCount;
  B->Used = true;
  return ErrorCode::Ok;
}

BufferQueue::ErrorCode BufferQueue::releaseBuffer(Buffer &Buf) {
  // Check whether the buffer being referred to is within the bounds of the
  // backing store's range.
  BufferRep *B = nullptr;
  {
    SpinMutexLock Guard(&Mutex);
    if (Buf.Data < BackingStore ||
        Buf.Data > reinterpret_cast<char *>(BackingStore) +
                       (BufferCount * BufferSize)) {
      if (Buf.Generation != generation()) {
        decRefCount(Buf.BackingStore, Buf.Size, Buf.Count);
        Buf.Data = nullptr;
        Buf.Size = 0;
        Buf.Generation = 0;
        Buf.Count = 0;
        Buf.BackingStore = nullptr;
        return BufferQueue::ErrorCode::Ok;
      }
      return BufferQueue::ErrorCode::UnrecognizedBuffer;
    }

    if (LiveBuffers == 0) {
      decRefCount(Buf.BackingStore, Buf.Size, Buf.Count);
      Buf.Data = nullptr;
      Buf.Size = Buf.Size;
      Buf.Generation = 0;
      Buf.BackingStore = nullptr;
      Buf.Count = 0;
      return ErrorCode::Ok;
    }

    --LiveBuffers;
    B = First++;
    if (First == (Buffers + BufferCount))
      First = Buffers;
  }

  // Now that the buffer has been released, we mark it as "used".
  B->Buff.Data = Buf.Data;
  B->Buff.Size = Buf.Size;
  B->Buff.Generation = Buf.Generation;
  B->Buff.BackingStore = Buf.BackingStore;
  B->Buff.Count = Buf.Count;
  B->Used = true;
  decRefCount(Buf.BackingStore, Buf.Size, Buf.Count);
  atomic_store(&B->Buff.Extents,
               atomic_load(&Buf.Extents, memory_order_acquire),
               memory_order_release);
  Buf.Data = nullptr;
  Buf.Size = 0;
  Buf.Generation = 0;
  Buf.BackingStore = nullptr;
  Buf.Count = 0;
  return ErrorCode::Ok;
}

BufferQueue::ErrorCode BufferQueue::finalize() {
  if (atomic_exchange(&Finalizing, 1, memory_order_acq_rel))
    return ErrorCode::QueueFinalizing;
  return ErrorCode::Ok;
}

void BufferQueue::cleanupBuffers() {
  for (auto B = Buffers, E = Buffers + BufferCount; B != E; ++B)
    B->~BufferRep();
  deallocateBuffer(Buffers, BufferCount);
  decRefCount(BackingStore, BufferSize, BufferCount);
  BackingStore = nullptr;
  Buffers = nullptr;
  BufferCount = 0;
  BufferSize = 0;
}

BufferQueue::~BufferQueue() { cleanupBuffers(); }
