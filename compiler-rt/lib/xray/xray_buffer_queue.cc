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

BufferQueue::BufferQueue(size_t B, size_t N,
                         bool &Success) XRAY_NEVER_INSTRUMENT
    : BufferSize(B),
      BufferCount(N),
      Mutex(),
      Finalizing{0},
      BackingStore(allocateBuffer(B *N)),
      Buffers(initArray<BufferQueue::BufferRep>(N)),
      Next(Buffers),
      First(Buffers),
      LiveBuffers(0) {
  if (BackingStore == nullptr) {
    Success = false;
    return;
  }
  if (Buffers == nullptr) {
    deallocateBuffer(BackingStore, BufferSize * BufferCount);
    Success = false;
    return;
  }

  for (size_t i = 0; i < N; ++i) {
    auto &T = Buffers[i];
    auto &Buf = T.Buff;
    Buf.Data = reinterpret_cast<char *>(BackingStore) + (BufferSize * i);
    Buf.Size = B;
    atomic_store(&Buf.Extents, 0, memory_order_release);
    T.Used = false;
  }
  Success = true;
}

BufferQueue::ErrorCode BufferQueue::getBuffer(Buffer &Buf) {
  if (atomic_load(&Finalizing, memory_order_acquire))
    return ErrorCode::QueueFinalizing;

  SpinMutexLock Guard(&Mutex);
  if (LiveBuffers == BufferCount)
    return ErrorCode::NotEnoughMemory;

  auto &T = *Next;
  auto &B = T.Buff;
  auto Extents = atomic_load(&B.Extents, memory_order_acquire);
  atomic_store(&Buf.Extents, Extents, memory_order_release);
  Buf.Data = B.Data;
  Buf.Size = B.Size;
  T.Used = true;
  ++LiveBuffers;

  if (++Next == (Buffers + BufferCount))
    Next = Buffers;

  return ErrorCode::Ok;
}

BufferQueue::ErrorCode BufferQueue::releaseBuffer(Buffer &Buf) {
  // Check whether the buffer being referred to is within the bounds of the
  // backing store's range.
  if (Buf.Data < BackingStore ||
      Buf.Data >
          reinterpret_cast<char *>(BackingStore) + (BufferCount * BufferSize))
    return ErrorCode::UnrecognizedBuffer;

  SpinMutexLock Guard(&Mutex);

  // This points to a semantic bug, we really ought to not be releasing more
  // buffers than we actually get.
  if (LiveBuffers == 0)
    return ErrorCode::NotEnoughMemory;

  // Now that the buffer has been released, we mark it as "used".
  auto Extents = atomic_load(&Buf.Extents, memory_order_acquire);
  atomic_store(&First->Buff.Extents, Extents, memory_order_release);
  First->Buff.Data = Buf.Data;
  First->Buff.Size = Buf.Size;
  First->Used = true;
  Buf.Data = nullptr;
  Buf.Size = 0;
  atomic_store(&Buf.Extents, 0, memory_order_release);
  --LiveBuffers;
  if (++First == (Buffers + BufferCount))
    First = Buffers;

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
