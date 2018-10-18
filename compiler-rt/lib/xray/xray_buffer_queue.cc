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
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_posix.h"
#include "xray_allocator.h"
#include "xray_defs.h"
#include <memory>
#include <sys/mman.h>

using namespace __xray;
using namespace __sanitizer;

BufferQueue::ErrorCode BufferQueue::init(size_t BS, size_t BC) {
  SpinMutexLock Guard(&Mutex);

  if (!finalizing())
    return BufferQueue::ErrorCode::AlreadyInitialized;

  bool Success = false;
  BufferSize = BS;
  BufferCount = BC;
  BackingStore = allocateBuffer(BufferSize * BufferCount);
  if (BackingStore == nullptr)
    return BufferQueue::ErrorCode::NotEnoughMemory;

  auto CleanupBackingStore = __sanitizer::at_scope_exit([&, this] {
    if (Success)
      return;
    deallocateBuffer(BackingStore, BufferSize * BufferCount);
  });

  Buffers = initArray<BufferRep>(BufferCount);
  if (Buffers == nullptr)
    return BufferQueue::ErrorCode::NotEnoughMemory;

  // At this point we increment the generation number to associate the buffers
  // to the new generation.
  atomic_fetch_add(&Generation, 1, memory_order_acq_rel);

  Success = true;
  for (size_t i = 0; i < BufferCount; ++i) {
    auto &T = Buffers[i];
    auto &Buf = T.Buff;
    atomic_store(&Buf.Extents, 0, memory_order_release);
    Buf.Generation = generation();
    Buf.Data = reinterpret_cast<char *>(BackingStore) + (BufferSize * i);
    Buf.Size = BufferSize;
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

  Buf.Data = B->Buff.Data;
  Buf.Generation = generation();
  Buf.Size = B->Buff.Size;
  B->Used = true;
  return ErrorCode::Ok;
}

BufferQueue::ErrorCode BufferQueue::releaseBuffer(Buffer &Buf) {
  // Check whether the buffer being referred to is within the bounds of the
  // backing store's range.
  if (Buf.Data < BackingStore ||
      Buf.Data >
          reinterpret_cast<char *>(BackingStore) + (BufferCount * BufferSize)) {
    if (Buf.Generation != generation()) {
      Buf.Data = nullptr;
      Buf.Size = 0;
      Buf.Generation = 0;
      return BufferQueue::ErrorCode::Ok;
    }
    return BufferQueue::ErrorCode::UnrecognizedBuffer;
  }

  BufferRep *B = nullptr;
  {
    SpinMutexLock Guard(&Mutex);

    // This points to a semantic bug, we really ought to not be releasing more
    // buffers than we actually get.
    if (LiveBuffers == 0)
      return ErrorCode::NotEnoughMemory;

    --LiveBuffers;
    B = First++;
    if (First == (Buffers + BufferCount))
      First = Buffers;
  }

  // Now that the buffer has been released, we mark it as "used".
  B->Buff.Data = Buf.Data;
  B->Buff.Size = Buf.Size;
  B->Buff.Generation = Buf.Generation;
  B->Used = true;
  atomic_store(&B->Buff.Extents,
               atomic_load(&Buf.Extents, memory_order_acquire),
               memory_order_release);
  Buf.Data = nullptr;
  Buf.Size = 0;
  Buf.Generation = 0;
  return ErrorCode::Ok;
}

BufferQueue::ErrorCode BufferQueue::finalize() {
  if (atomic_exchange(&Finalizing, 1, memory_order_acq_rel))
    return ErrorCode::QueueFinalizing;
  return ErrorCode::Ok;
}

BufferQueue::~BufferQueue() {
  for (auto B = Buffers, E = Buffers + BufferCount; B != E; ++B)
    B->~BufferRep();
  deallocateBuffer(Buffers, BufferCount);
  deallocateBuffer(BackingStore, BufferSize * BufferCount);
}
